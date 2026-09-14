# Setup Guide

## Prerequisites

- Python 3.11+ (the API image runs 3.12)
- [uv](https://docs.astral.sh/uv/) package manager
- Node 22 + [pnpm](https://pnpm.io/) for the web UI
- GPU recommended for embedding extraction (CPU works, but slow)
- Docker for containerized deployment

All commands below run from the repo root unless stated otherwise. Run
`make help` for the full target list.

## Data

Unzip `altur-challenge-audio.zip` into `apps/train/data/audio/` (gitignored):

```text
apps/train/data/
├── audio/          # <anon_id>.wav — stereo 8 kHz PCM16, gitignored
├── manifest.csv    # anon_id, label, split, duration_s
└── turns/          # per-call speech segments
```

`manifest.csv` and `turns/` are committed. Augmentation writes generated audio to
`apps/train/data/augmented_train/`.

## Install

```bash
make sync          # uv sync for apps/api, apps/dataset, apps/train and tests
```

Or per project:

```bash
uv sync --project apps/api
uv sync --project apps/dataset
uv sync --project apps/train
uv sync --project tests
```

## Training Pipeline

Modes accepted by `python -m src <mode> --config <config>`:

| Mode | Effect |
| ---- | ------ |
| `extract` | Cache WavLM embeddings for the base train/val splits |
| `extract_augmented` | Cache embeddings for the augmented train split (round two) |
| `train` | Train the MLP from cached embeddings |
| `round2` | `extract_augmented` + fine-tune from the baseline checkpoint |
| `calibrate` | Fit the isotonic calibrator on val predictions |
| `all` | `extract` + `train` (no calibration) |

### 1. Extract embeddings

```bash
make extract       # == make train TRAIN_MODE=extract
```

- Runs frozen WavLM-large over every window of every call
- Saves `checkpoints/embeddings_cache/{train,val}.pt`
- ~30 min on GPU; skips splits whose cache file already exists
- Requires `apps/train/data/audio/`

### 2. Train the classifier

```bash
make train TRAIN_MODE=train
```

- Loads cached embeddings, trains the MLP for up to 30 epochs
- Early stopping on validation EER (patience 7)
- Saves `checkpoints/best_model.pt`
- Seconds from cache, not hours

### 3. Fit the calibrator

```bash
make calibrate
```

- Fits `sklearn.IsotonicRegression` on validation predictions
- Saves `checkpoints/calibrator.joblib`
- Prints EER and Brier score before and after calibration

`make train` without `TRAIN_MODE` runs the all-in-one `extract` + `train` path;
it does **not** fit the calibrator.

### Optional: round two (telephony augmentation)

```bash
make round2            # augment train split, extract, fine-tune from baseline
make round2-calibrate  # fit the round-two calibrator
```

- Augmentation is deterministic (seed 2026, 3 variants per train call) and never
  touches validation audio; see `apps/dataset/README.md`
- Round-two outputs land in `checkpoints/round2/`, not `checkpoints/`
- The API loads `checkpoints/best_model.pt` + `checkpoints/calibrator.joblib`, so
  promote round-two weights by copying them there or by setting `model_path` and
  `calibrator_path` (see [Environment Variables](#environment-variables))

### Embedding cache

`checkpoints/embeddings_cache/` holds `train.pt` (20,378 windows), `val.pt`
(5,199 windows) and, after round two, `train_augmented.pt`. Delete a file to
force re-extraction for that split.

## API (Local)

```bash
make api
# == cd apps/api && PYTHONPATH=src uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

`PYTHONPATH=src` is required: `apps/api/pyproject.toml` sets `package = false`,
so `src/` is not on `sys.path` automatically.

Models load **eagerly at startup** via the `@app.on_event("startup")` hook.
`GET /health` reports `device`, `dtype` and `ready`:

```bash
curl -s localhost:8000/health
# {"status":"ok","device":"cpu","dtype":"float32","ready":true}
```

Local runs also need `wavlm-large.onnx` (see [Benchmarks](#benchmarks)); without
it the API still starts but reports `ready: false` and `/detect` answers 503.

## Web UI (Local)

```bash
cd apps/web && pnpm install && pnpm dev
```

Serves `http://localhost:3000`, reads the API at `NEXT_PUBLIC_API_URL`
(default `http://localhost:8000`), shows `/health` state and lets you drop a WAV
to see the verdict and confidence.

## API + Web (Docker)

Dev — builds both images locally:

```bash
docker compose -f docker-compose.dev.yml up --build
```

Prod — pulls the pinned GHCR images:

```bash
docker compose -f docker-compose.prod.yml up
```

The API image exports WavLM-large to ONNX at build time and copies
`apps/train/checkpoints/` into `/app/model/checkpoints/`, so the runtime needs no
`transformers` and no GPU libraries. `.github/workflows/build.yml` pushes
`api` and `web` images with `latest` and commit-SHA tags on every push to `main`.

## Testing

```bash
make test          # dataset package tests + everything under tests/
make test-dataset  # only apps/dataset/tests
make lint          # ruff check
make format        # ruff format
make check         # lint + test
```

Targeted runs:

```bash
uv run --project tests pytest tests/unit -v           # inference policy (no API needed)
uv run --project tests pytest tests/integration -v    # needs a live API + audio
```

Integration tests (tests/integration/test_detect.py) cover the health endpoint,
happy-path detection, val-set integrity, error cases (bad base64, empty WAV,
missing fields) and response-contract shape. They require `apps/train/data/audio/`
and a running endpoint; override `API_URL`, `AUDIO_DIR` and `MANIFEST` to point
elsewhere.

## Judge Harness

`tests/judge/check_endpoint.py` posts dataset calls to `/detect` exactly like the
judge does and scores the answers. Standard library only:

```bash
python tests/judge/check_endpoint.py \
  --url http://localhost:8000/detect \
  --manifest apps/train/data/manifest.csv \
  --audio-dir apps/train/data/audio \
  --split val
```

Fixture builders for robustness testing (require the `datasets` package from the
`tests` project):

| Script / target | Builds |
| --------------- | ------ |
| `tests/judge/build_test_set.py` | Balanced ASVspoof 2019 LA subset as 8 kHz stereo WAVs + manifest |
| `tests/judge/build_spanish_test_set.py` | HABLA Spanish clips joined into ≥60 s calls |
| `make augment-judge` | Deterministic perturbations of val audio in `tests/judge/augmented_val/` (`split=hidden`) |

## Benchmarks

`tests/benchmark/inference_latency.py` (per-stage + E2E),
`tests/benchmark/onnx_comparison.py` (ONNX-only) and
`tests/benchmark/pytorch_vs_onnx.py` (backend comparison).

They need `wavlm-large.onnx`, which the API Docker build produces and the repo
does not commit. Export it the same way the Dockerfile does:

```bash
cd apps/train
uv run --with onnx python -c "
import torch
from transformers import WavLMModel
m = WavLMModel.from_pretrained('microsoft/wavlm-large').eval()
with torch.no_grad():
    torch.onnx.export(
        m, torch.randn(1, 48000), 'checkpoints/wavlm-large.onnx',
        input_names=['input_values'], output_names=['last_hidden_state'],
        dynamic_axes={'input_values': {0: 'batch'}, 'last_hidden_state': {0: 'batch'}},
        opset_version=14, dynamo=False,
    )
"
```

Then run from the repo root with the API environment active (it provides
`onnxruntime`):

```bash
source apps/api/.venv/bin/activate
python -m tests.benchmark.inference_latency
```

Measured on an AMD Ryzen 5 9600X (6C/12T, CPU-only, 4 intra-op threads): WavLM extraction ~170 ms mean and always ~90% of the
end-to-end latency budget (99% of in-process stage time), 0.17-0.24 s end-to-end
across 2-20 s calls, p95 ≤ 0.28 s. Against the
deployed endpoint the average is ~0.5-0.6 s per call. See
[benchmarks.md](benchmarks.md).

## Environment Variables

The API reads `apps/api/src/.env` if present (see `.env.example`); every value can
also be passed as an environment variable. Names match the `Settings` fields and
are matched case-insensitively (`max_windows` or `MAX_WINDOWS`).

| Variable | Default | Description |
| ---------- | --------- | ------------- |
| `device` | `cuda` if available, else `cpu` | Inference device |
| `dtype` | `float16` on CUDA, else `float32` | Model precision |
| `max_windows` | `1` | Highest-energy caller windows scored per call |
| `onnx_intra_threads` | `4` | ONNX intra-op CPU threads (spinning disabled) |
| `decision_threshold` | `0.15` | Synthetic decision threshold, recentered to 0.5 |
| `extract_batch_size` | `16` | Batch size for WavLM extraction |
| `model_path` | auto-resolved | Path to `best_model.pt` |
| `calibrator_path` | auto-resolved | Path to `calibrator.joblib` |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | Only read by `python -m` in `apps/api/src` |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Web build arg / runtime URL for the API |
| `HUGGING_FACE_HUB_TOKEN` | unset | Optional token for the WavLM download during image build |

`CORS_ORIGINS` was removed: the API now allows every origin without credentials so
the bundled UI and judge tooling can post directly.

### Model path resolution

An explicit `model_path` / `calibrator_path` wins (relative paths resolve against
`apps/api/src/`). Otherwise the first existing candidate is used:

1. `apps/train/checkpoints/` (local repo layout)
2. `model/checkpoints/` relative to the API source dir (`/app/model/checkpoints` in Docker)
3. `train/checkpoints/` relative to the API source dir (sibling layout)

`wavlm-large.onnx` is resolved separately by walking up from
`apps/api/src/models/extractor.py`, checking `checkpoints/` and
`model/checkpoints/` at each level and in sibling directories.

## Troubleshooting

### "Model not found" on API startup

- Ensure `checkpoints/best_model.pt` and `calibrator.joblib` exist in one of the
  resolved paths above, or set `model_path` / `calibrator_path` explicitly
- Check `GET /health` for `ready`, `device` and `dtype`

### `ready: false` and 503 on `/detect` locally

- `wavlm-large.onnx` is missing. Export it (see [Benchmarks](#benchmarks)) or run
  the container, which builds it during the image build

### `InconsistentVersionWarning` when loading the calibrator

- The committed `calibrator.joblib` was fitted with scikit-learn 1.6.1 while the
  runtime has 1.9.1. It is a pickle-version warning, not a failure. Re-run
  `make calibrate` to refit with the installed version

### Slow first request

- Models load at startup, so the first *request* is already warm; container start
  takes a few seconds (ONNX session creation, ~2.7 s on a 6-core CPU)

### CUDA out of memory during extraction

- Reduce `extract_batch_size` in the config (default 32, try 16 or 8)
- Embedding extraction is the only GPU-intensive step

### Port already in use

- `make api API_PORT=8001`, or stop the container holding port 8000
