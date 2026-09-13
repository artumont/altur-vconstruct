# Setup Guide

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- GPU recommended for training (CPU works but slow)
- Docker (for deployment)

## Training Pipeline

All commands from repo root:

```bash
cd apps/train
uv sync
```

### 1. Extract Embeddings

```bash
uv run python -m src extract --config configs/baseline.yml
```

- Runs WavLM-large over all audio files
- Saves to `checkpoints/embeddings_cache/{split}.pt`
- Takes ~30 min on GPU, skips if cache exists
- Requires `audio/` directory with WAV files (unzipped from altur-challenge-audio.zip)

### 2. Train Classifier

```bash
uv run python -m src train --config configs/baseline.yml
```

- Loads cached embeddings
- Trains MLP for up to 30 epochs with early stopping
- Saves best model to `checkpoints/best_model.pt`
- Takes seconds (embeddings are pre-computed)

### 3. Fit Calibrator

```bash
uv run python -m src calibrate --config configs/baseline.yml
```

- Fits isotonic regression on validation predictions
- Saves to `checkpoints/calibrator.joblib`
- Reports pre/post EER and Brier score

### All-in-one

```bash
uv run python -m src all --config configs/baseline.yml
```

Runs extract + train in sequence.

## API (Local)

```bash
cd apps/api
uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Models load lazily on first request (or eagerly via `GET /health`).

## API (Docker)

From repo root:

```bash
docker compose up --build
```

This builds the API image with:

- WavLM-large exported to ONNX at build time
- Trained classifier weights copied from `apps/train/checkpoints/`
- Lean runtime (no transformers, no GPU libs)

API available at `http://localhost:8000`.

## Testing

### Integration Tests

```bash
cd tests/integration
uv sync
# Assumes API is running (docker compose up or uvicorn)
uv run pytest -v
```

Tests:

- Health endpoint
- Happy path (synthetic/human detection)
- Val dataset integrity (real audio samples)
- Error cases (bad base64, empty WAV, missing fields)
- Contract compliance (response shape, confidence bounds)

### Benchmarks

```bash
cd apps/api
uv run python -m tests.benchmark.onnx_comparison
```

Reports per-stage latency across audio durations.

## Environment Variables

| Variable | Default | Description |
| ---------- | --------- | ------------- |
| `device` | `cuda` / `cpu` | Inference device |
| `dtype` | `float16` / `float32` | Model precision |
| `max_windows` | `1` | Highest-energy caller windows scored per call |
| `onnx_intra_threads` | `4` | ONNX intra-op CPU threads |
| `decision_threshold` | `0.15` | Synthetic probability decision threshold |
| `extract_batch_size` | `16` | Batch size for WavLM extraction |
| `model_path` | auto-resolved | Path to `best_model.pt` |
| `calibrator_path` | auto-resolved | Path to `calibrator.joblib` |

Model path resolution order:

1. Explicit `model_path` env var
2. `apps/train/checkpoints/` (local repo)
3. `/app/model/checkpoints/` (Docker image)
4. `apps/api/train/checkpoints/` (sibling layout)

## Data Layout

```
apps/train/
  data/
    audio/            # WAV files (gitignored, unzip from release)
      <anon_id>.wav
    manifest.csv      # anon_id, label, split, duration_s
    turns/            # Per-call JSON with speech segments
  checkpoints/
    best_model.pt     # Trained classifier weights
    calibrator.joblib # Isotonic regression calibrator
    embeddings_cache/
      train.pt        # Cached train embeddings
      val.pt          # Cached val embeddings
```

## Troubleshooting

**"Model not found" on API startup**

- Ensure `checkpoints/best_model.pt` exists in one of the resolved paths
- Check `GET /health` for device/status info

**Slow first request**

- Models load on first request if not pre-loaded
- Use `GET /health` to trigger eager loading
- Docker image pre-downloads WavLM at build time

**CUDA out of memory during extraction**

- Reduce `extract_batch_size` in config (default 32, try 16 or 8)
- Embedding extraction is the only GPU-intensive step
