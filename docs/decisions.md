# Design Decisions

Key architectural and engineering choices made during development, with rationale.

## 1. High-Energy Window Sampling for Latency

**Problem**: A 3-minute call produces ~119 overlapping 3-second inference windows at 50% hop. Scoring all of them is the dominant latency cost because WavLM extraction is expensive per window.

**Decision**: Score one highest-energy caller window at inference (`max_windows=1`). Training continues using all 4-second windows.

**Rationale**: One window cuts encoder work to a fixed cost independent of call duration. Selecting by caller-channel energy avoids greetings or silence consuming the only extraction. Mean pooling keeps the 1024-dimensional classifier interface unchanged when inference windows shrink from four to three seconds.

**Impact**: On 213 augmented validation calls, this policy retained 99.53% accuracy and measured 0.999s mean HTTP latency on an unplugged Ryzen 5 PRO 4650U. Short calls under three seconds use one padded window. `sample_windows()` implements selection in `apps/api/src/pipeline/service.py`.

## 2. ONNX Runtime Instead of PyTorch for Inference

**Problem**: WavLM-large (~300M params) loaded via `transformers` + PyTorch is slow on CPU and pulls heavy dependencies (full PyTorch stack, transformers, tokenizers).

**Decision**: Export WavLM to ONNX at Docker build time. Run inference via `onnxruntime`.

**Rationale**:

- ~2x faster on CPU (graph optimization, no Python overhead)
- No `transformers` dependency in the API image
- Smaller runtime footprint
- Same interface — `ONNXSSLEvaluator` is a drop-in replacement for `SSLEvaluator`

**Implementation**: `torch.onnx.export` in the Dockerfile with opset 14, dynamic batch axis. The ONNX model is searched across multiple candidate paths (Docker layout vs local repo layout) in `_default_onnx_path()`.

## 3. Two-Layer Architecture (SSL + MLP)

**Alternatives considered**:

- End-to-end fine-tuning of WavLM
- Complex architectures (AASIST, conformers, attention pooling)
- XGBoost on hand-crafted features (MFCC, LFCC)

**Decision**: Frozen SSL backbone + 3-layer MLP head.

**Rationale**:

- Only 282 training calls — fine-tuning 300M params risks severe overfitting
- Frozen WavLM embeddings are already high-quality features for anti-spoofing (2.29% EER on ASVspoof)
- MLP trains in seconds — fast iteration cycle
- No heavy augmentation needed to prevent overfitting

**Trade-off**: Sacrifices potential accuracy from end-to-end training in exchange for generalization on small data and fast development velocity.

## 4. Isotonic Calibration Over Platt Scaling

**Problem**: Raw sigmoid outputs from the MLP are not reliable probabilities. The challenge explicitly uses `confidence` for tie-breaking — judges trust well-calibrated scores.

**Alternatives considered**:

- Platt scaling (logistic regression on logits)
- Temperature scaling
- No calibration (use raw sigmoid)

**Decision**: Fit `sklearn.IsotonicRegression` on validation predictions.

**Rationale**:

- Non-parametric — no assumption about score distribution shape
- Handles the bimodal distribution (human scores cluster near 0, synthetic near 1) better than Platt
- Simple to fit and serialize (`calibrator.joblib`)
- Loaded once at API startup, adds negligible latency

## 5. Pre-computed Embeddings Cache

**Problem**: WavLM inference takes ~30 minutes for 353 calls on GPU. Every config change requiring re-training means waiting 30+ minutes just for extraction.

**Decision**: Extract once, save to `{split}.pt` files. Subsequent runs load from cache.

**Implementation**:

- `pre_extract_embeddings()` in `apps/train/src/train.py` skips already-cached splits
- Cache format: `{"embeddings": Tensor(N, 1024), "labels": Tensor(N)}`
- Stored in `apps/train/checkpoints/embeddings_cache/`

**Impact**: Training iteration goes from ~35 minutes to ~5 seconds.

## 6. Speaker-Disjoint Split

**Context**: The dataset comes pre-split by Altur. No caller appears in both train and val.

**Why this matters**: If the same speaker appears in both splits, the model can memorize speaker identity (pitch, accent, recording quality) instead of learning synthetic vs human vocal characteristics. Speaker-disjoint splits force the model to generalize across unseen speakers.

**Decision**: Use the provided split as-is. No mixing or re-splitting.

## 7. Channel Separation (Caller vs Agent)

**Context**: Stereo WAV contains both parties. Channel 0 is always the caller.

**Decision**: Extract only channel 0 (caller) for classification. Channel 1 (agent) is ignored.

**Rationale**: The task is to classify the caller, not the agent. The agent's voice is constant across calls and adds noise.

**Future enhancement**: Conversational features from both channels (response latency, interruption patterns, silence behavior) could improve detection. The `data/turns/` JSONs already contain per-call speech segments that could support this.

## 8. Docker Multi-Stage Build

**Problem**: Full PyTorch + transformers stack is large (~2GB+). Shipping it in the runtime image wastes resources and slows deployment.

**Decision**: Two-stage Docker build:

1. **Builder**: installs deps, downloads WavLM-large from HuggingFace, exports to ONNX
2. **Runtime**: copies only the ONNX model + trained MLP weights + ONNX Runtime

**Rationale**:

- Runtime image is lean (no `transformers`, no GPU libs)
- ONNX export happens once at build time, not on every container start
- `HF_HOME` cache in builder layer avoids re-downloading on rebuilds
- Model paths resolved dynamically (`config.py` searches Docker layout vs local layout)

## 9. Training Hyperparameters

Chosen via small-scale sweeps on the validation set:

| Parameter | Value | Notes |
| ----------- | ------- | ------- |
| Learning rate | 1e-4 | AdamW default works well for small data |
| Weight decay | 1e-4 | Light regularization |
| Batch size | 64 | Fits in GPU memory, stable gradients |
| Max epochs | 30 | Early stopping usually triggers before this |
| Patience | 7 | Enough to escape local minima, not too long |
| Dropout | 0.3 | Prevents overfitting on small dataset |
| Scheduler | Cosine annealing | Smooth LR decay |

## 10. Endpoint Contract Compliance

**Decision**: Strict adherence to the challenge contract:

```json
{
  "is_synthetic": true,
  "confidence": 0.87
}
```

- `confidence` rounded to 4 decimal places
- `is_synthetic` is a boolean (not a string, not a number)
- HTTP 400 for bad audio, 503 for missing model
- Health endpoint at `GET /health` for monitoring

Models are pre-loaded at startup (`@app.on_event("startup")`) so the first request is fast — latency is a scoring criterion.
