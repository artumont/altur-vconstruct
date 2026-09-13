# Architecture

## Overview

Two-stage pipeline: frozen SSL backbone extracts embeddings, lightweight MLP classifies.

```
Audio (8 kHz stereo WAV)
  |
  +-- Extract channel 0 (caller)
  |
  +-- Resample 8 kHz -> 16 kHz
  |
  +-- Window into 3s inference chunks (50% overlap)
  |
  +-- Select one highest-energy caller window  <-- latency optimization
  |
  +-- WavLM-large (frozen) -> 1024-dim embeddings (mean-pooled)
  |
  +-- MLP classifier -> spoof probability per window
  |
  +-- Isotonic calibration + threshold recentering
  |
  +-- { is_synthetic, confidence }
```

## SSL Backbone: WavLM-large

`microsoft/wavlm-large` — 300M parameter self-supervised speech model.

**Why WavLM over alternatives:**

| Backbone | EER (ASVspoof 2021 LA) |
| ---------- | ---------------------- |
| **WavLM-large** | **2.29%** |
| XLS-R-300M + AASIST | 2.65-2.84% |
| Wav2Vec2-large | 3.09% |

WavLM achieves the lowest error rate on standard anti-spoofing benchmarks. Its frozen embeddings transfer well to small datasets without fine-tuning the backbone — critical here with only 282 training calls.

The backbone is **fully frozen** during training. We extract 1024-dim embeddings by mean-pooling the `last_hidden_state` over the time axis:

```
Training input:  (B, 64000) — 4s mono at 16 kHz
Inference input: (B, 48000) — 3s mono at 16 kHz
Output:          (B, 1024)  — mean-pooled embedding
```

## Classifier: SpoofClassifier MLP

3-layer MLP with BatchNorm and Dropout:

```
Linear(1024 -> 256) -> BatchNorm1d -> ReLU -> Dropout(0.3)
Linear(256 -> 128)  -> BatchNorm1d -> ReLU -> Dropout(0.3)
Linear(128 -> 1)    -> Sigmoid
```

~133K parameters. Trained with BCE loss, AdamW optimizer (lr=1e-4, weight_decay=1e-4), cosine annealing schedule. Early stopping on EER with patience=7.

**Why MLP over complex architectures (AASIST, conformers):**

- Small dataset (282 calls) — deep architectures overfit
- Frozen embeddings are already high-quality features
- Training takes seconds, not hours — fast iteration

## Calibration

Raw sigmoid outputs are not reliable probabilities. The challenge uses `confidence` for tie-breaking, so calibration matters.

We fit `sklearn.IsotonicRegression` on validation-set predictions. Isotonic is non-parametric (no sigmoid shape assumption) and handles the bimodal score distribution better than Platt scaling.

The fitted calibrator is saved as `calibrator.joblib` and loaded at API startup. At inference, its synthetic probability is recentered so the configured `decision_threshold=0.15` maps to response boundary 0.5 while preserving score ranking.

## Channel Separation

Stereo WAV contains both parties:

- **Channel 0**: caller (what we classify)
- **Channel 1**: agent (context, not used in current architecture)

The resampler extracts channel 0 as mono and resamples from 8 kHz to 16 kHz (WavLM's expected sample rate) via `torchaudio.transforms.Resample`.

Channel 1 is not currently fed to the classifier. The architecture classifies the caller's **acoustic properties** only. Conversational features (response latency, interruption patterns) are documented as a potential enhancement in `docs/decisions.md`.

## Embedding Cache

WavLM inference is the bottleneck during training (~30 min for 353 calls on GPU). Embeddings are extracted once and cached:

- `checkpoints/embeddings_cache/train.pt` — (N_train, 1024) tensor + labels
- `checkpoints/embeddings_cache/val.pt` — (N_val, 1024) tensor + labels

Subsequent training runs load from cache in seconds. The `pre_extract_embeddings()` function skips already-cached splits.

## Two Runtime Paths

| Context | Extractor | Purpose |
|---------|-----------|---------|
| Training | PyTorch `WavLMModel` | GPU extraction, batch processing |
| Inference | ONNX `ONNXSSLEvaluator` | CPU-optimized, ~2x faster |

Both implement the same interface (`extract`, `extract_batch`). The ONNX path is used in the API; the PyTorch path is used during training.
