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

~296K parameters (Linear(1024,256) is 262,400 of them). Trained with BCE loss, AdamW optimizer (lr=1e-4, weight_decay=1e-4), cosine annealing schedule. Early stopping on EER with patience=7.

**Why MLP over complex architectures (AASIST, conformers):**

- Small dataset (282 calls) — deep architectures overfit
- Frozen embeddings are already high-quality features
- Training takes seconds, not hours — fast iteration

## Calibration

Raw sigmoid outputs are not reliable probabilities. The challenge uses `confidence` for tie-breaking, so calibration matters.

We fit `sklearn.IsotonicRegression` on validation-set predictions. Isotonic is non-parametric (no sigmoid shape assumption) and handles the bimodal score distribution better than Platt scaling.

The fitted calibrator is saved as `calibrator.joblib` and loaded at API startup. At inference, its synthetic probability is recentered so the configured `decision_threshold=0.15` maps to response boundary 0.5 while preserving score ranking.

## Training Pipeline

Two rounds, both training only the MLP head on frozen WavLM embeddings
(`apps/train/src/train.py`):

1. **Baseline** (`configs/baseline.yml`) — every 4-second window of the train
   split. AdamW (lr `1e-4`, weight decay `1e-4`), batch 64, cosine annealing,
   early stopping on validation EER with patience 7, up to 30 epochs.
2. **Round two** (`configs/round2.yml`, optional) — fine-tune the baseline
   checkpoint at lr `1e-5` for up to 15 epochs (patience 4) on the baseline
   train embeddings **plus** `train_augmented.pt`.

Augmentation lives in `apps/dataset/src/dataset/augment.py` and is deterministic
for a fixed seed (`make round2` uses seed 2026 and 3 variants per train call,
i.e. 846 variants from the 282 train calls, taking the training pool to ~800+
augmented calls). Each generated variant applies one telephony-style family:
caller/agent latency shift (100-1500 ms), call-level gain (-5..+5 dB) with
18-30 dB SNR noise, or packet-loss dropout (2-8 gaps) plus 0.97-1.03x clock-rate
variation. Labels are never changed, and the validation split is never augmented
for training.

## Shipped Checkpoint

`apps/train/checkpoints/best_model.pt` + `calibrator.joblib` are what the API
loads. Scored read-only over the cached validation windows (all 4-second training
windows, 5,199 windows from the 71 held-out val calls):

| Metric | Value |
| ------ | ----- |
| EER, raw sigmoid | 0.40% |
| EER, post-isotonic | 0.44% |
| Window accuracy at recentered threshold | 99.5% (5,174 / 5,199) |
| Brier, post-isotonic | 0.0039 |

The endpoint scores one 3-second high-energy window per call, so these per-window
figures are a proxy for the judge metric rather than the exact number. Measured
cost of that single window on an AMD Ryzen 5 9600X (6C/12T, CPU-only, 4 ONNX
intra-op threads): WavLM extraction ~170 ms mean and always ~90% of the
end-to-end latency budget (99% of in-process pipeline
time), 0.17-0.24 s end-to-end for 2-20 s calls, p95 <= 0.28 s. On deployment
hardware the endpoint averages ~0.5-0.6 s per call, down from ~1.0-1.2 s for the
earlier pipeline that scored every window without thread tuning. See
[benchmarks.md](benchmarks.md) for the full breakdown.

## Channel Separation

Stereo WAV contains both parties:

- **Channel 0**: caller (what we classify)
- **Channel 1**: agent (context, not used in current architecture)

The resampler extracts channel 0 as mono and resamples from 8 kHz to 16 kHz (WavLM's expected sample rate) via `torchaudio.transforms.Resample`.

Channel 1 is not currently fed to the classifier. The architecture classifies the caller's **acoustic properties** only. Conversational features (response latency, interruption patterns) are documented as a potential enhancement in `docs/decisions.md`.

## Embedding Cache

WavLM inference is the bottleneck during training (~30 min for 353 calls on GPU). Embeddings are extracted once and cached:

- `checkpoints/embeddings_cache/train.pt` — (20,378, 1024) tensor + labels
- `checkpoints/embeddings_cache/val.pt` — (5,199, 1024) tensor + labels
- `checkpoints/embeddings_cache/train_augmented.pt` — added when round two runs

Subsequent training runs load from cache in seconds. The `pre_extract_embeddings()` function skips already-cached splits.

## Two Runtime Paths

| Context | Extractor | Purpose |
|---------|-----------|---------|
| Training | PyTorch `WavLMModel` | GPU extraction, batch processing |
| Inference | ONNX `ONNXSSLEvaluator` | CPU-optimized, ~2x faster |

Both implement the same interface (`extract`, `extract_batch`). The ONNX path is used in the API; the PyTorch path is used during training.
