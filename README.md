# hackmty26 — Voice Anti-Spoofing for Altur Challenge

Detect whether the caller in a phone call is a **real human** or a **synthetic voice** (TTS deepfake).

Built for the [HackMTY 2026 Altur Challenge](https://hackmty.com/).

## Challenge

Given a stereo WAV recording, classify the caller (channel 0) as human or synthetic.

```
POST /detect
  Body:  {"audio_base64": "<base64 stereo 8 kHz WAV>"}
  Resp:  {"is_synthetic": true/false, "confidence": 0.0–1.0}
```

`confidence` breaks ties. Latency is a scoring criterion. Judge runs a hidden test set against the live endpoint.

## Dataset

353 recorded calls, split speaker-disjoint:

| Split | Calls | Human | Synthetic | Avg Duration |
|-------|-------|-------|-----------|-------------|
| Train | 282   | 113   | 169       | ~148s       |
| Val   | 71    | 37    | 34        | ~149s       |

Audio: 8 kHz telephony, 16-bit PCM, stereo. Per-call turn JSONs in `data/turns/`.

## Architecture

```
Audio (8 kHz stereo WAV)
  -> Resample caller channel to 16 kHz
  -> Window into 4s chunks (50% overlap)
  -> Sample max 2 windows (latency optimization)
  -> WavLM-large (frozen) -> 1024-dim embeddings
  -> MLP classifier -> isotonic calibration
  -> { is_synthetic, confidence }
```

- **Backbone**: `microsoft/wavlm-large` (frozen, 2.29% EER on ASVspoof 2021)
- **Classifier**: 3-layer MLP (1024 -> 256 -> 128 -> 1) with BatchNorm + Dropout
- **Calibration**: Isotonic regression for well-calibrated confidence scores
- **Inference**: ONNX Runtime (~2x faster than PyTorch on CPU)

## Quick Start

```bash
# Train (from repo root)
cd apps/train && uv sync
uv run python -m src all --config configs/baseline.yml

# Run API
cd ../api && uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 8000

# Or via Docker
cd ../.. && docker compose up --build
```

## Project Structure

```
hackmty26/
├── apps/
│   ├── api/               # FastAPI inference service (POST /detect)
│   └── train/             # Training pipeline (extract, train, calibrate)
├── tests/
│   ├── integration/       # HTTP endpoint tests
│   └── benchmark/         # ONNX vs PyTorch latency comparison
├── data/
│   ├── manifest.csv       # anon_id, label, split, duration_s
│   └── turns/             # Per-call speech segment JSONs
├── docs/                  # Detailed documentation
└── docker-compose.yml
```

## Documentation

| Doc | Description |
| ----- | ------------- |
| [docs/architecture.md](docs/architecture.md) | Model design, SSL backbone, MLP, calibration |
| [docs/decisions.md](docs/decisions.md) | All design decisions with rationale |
| [docs/inference.md](docs/inference.md) | Inference pipeline, latency, benchmarking |
| [docs/setup.md](docs/setup.md) | Training, API, Docker, testing, env vars |
| [docs/benchmarks.md](docs/benchmarks.md) | PyTorch vs ONNX comparison, per-stage latency |

## Tech Stack

| Component | Choice |
| ----------- | -------- |
| SSL backbone | WavLM-large |
| Inference | ONNX Runtime |
| Classifier | MLP (1024->256->128->1) |
| Calibration | Isotonic regression |
| API | FastAPI + uvicorn |
| Deployment | Docker multi-stage |
| Package manager | uv |
