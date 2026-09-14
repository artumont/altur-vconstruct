# altur-vconstruct

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

`GET /health` reports the loaded device/dtype and readiness (used by the Docker healthcheck and the web UI). The API allows any CORS origin without credentials so the bundled UI and judge tooling can call it directly.

## Dataset

353 recorded calls, split speaker-disjoint:

| Split | Calls | Human | Synthetic | Avg Duration |
|-------|-------|-------|-----------|-------------|
| Train | 282   | 113   | 169       | ~148s       |
| Val   | 71    | 37    | 34        | ~149s       |

Audio: 8 kHz telephony, 16-bit PCM, stereo. Per-call turn JSONs in `apps/train/data/turns/`. WAVs live in `apps/train/data/audio/` (gitignored, unzip `altur-challenge-audio.zip` there).

Training does not stop at 282 calls: `apps/dataset` generates 3 deterministic telephony variants per train call (846 variants, so ~800+ augmented training calls) so accuracy holds up under noise, gain, packet-loss and caller/agent latency shifts. Validation audio is never augmented.

## Architecture

```
Audio (8 kHz stereo WAV)
  -> Resample caller channel to 16 kHz
  -> Window into 3s chunks (50% overlap)
  -> Select highest-energy caller window (latency optimization)
  -> WavLM-large (frozen) -> 1024-dim embeddings
  -> MLP classifier -> isotonic calibration
  -> { is_synthetic, confidence }
```

- **Backbone**: `microsoft/wavlm-large` (frozen, 2.29% EER on ASVspoof 2021)
- **Classifier**: 3-layer MLP (1024 -> 256 -> 128 -> 1) with BatchNorm + Dropout
- **Calibration**: Isotonic regression for well-calibrated confidence scores
- **Inference**: ONNX Runtime (~2x faster than PyTorch on CPU), 4 intra-op threads

## Measured Results

Shipped checkpoint (`apps/train/checkpoints/best_model.pt` + `calibrator.joblib`),
scored read-only over the cached validation windows (5,199 windows from the 71
held-out val calls) — no retraining involved:

| Metric | Value |
| ------ | ----- |
| EER (raw sigmoid) | 0.40% |
| EER (post-isotonic) | 0.44% |
| Window accuracy @ recentered threshold | 99.5% (5,174 / 5,199) |
| Brier (post-isotonic) | 0.0039 |
| `/detect` end-to-end, average | ~0.5-0.6 s (~1.0-1.2 s before optimization) |

Validation windows are 4-second training windows; the endpoint scores one 3-second high-energy window, so these are a proxy for the judge metric, not the  exact number. The ~1.0-1.2 s figure is the earlier pipeline that scored every window without window selection or ONNX thread tuning; selecting a single  highest-energy window plus tuning ONNX CPU threads brought the average toabout 0.5-0.6 s. In-process stage timing on a 6-core desktop CPU puts the WavLM pass  at about 170 ms and the whole pipeline at 0.17-0.24 s for 2-20 s calls, so hardware and container overhead swing this more than the weights do. Re-run  `tests/benchmark/inference_latency.py` on deployment hardware before quoting a number.

## Quick Start

Everything is driven from the repo root
[Makefile](Makefile) (`make help` lists all targets):

```bash
make sync                 # install every project (api, dataset, train, tests)

# Baseline training — needs apps/train/data/audio/ unzipped first
make extract              # cache WavLM embeddings (GPU recommended)
make train TRAIN_MODE=train
make calibrate            # fit isotonic calibrator on val predictions

# Optional round two — deterministic telephony augmentation + fine-tune
make round2
make round2-calibrate

make api                  # PYTHONPATH=src uvicorn app:app --port 8000
```

Web UI:

```bash
cd apps/web && pnpm install && pnpm dev
```

Docker (from repo root):

```bash
docker compose -f docker-compose.dev.yml up --build   # build API + web locally
docker compose -f docker-compose.prod.yml up          # pinned GHCR images
```

## Documentation

| Doc | Description |
| ----- | ------------- |
| [docs/architecture.md](docs/architecture.md) | Model design, SSL backbone, MLP, calibration |
| [docs/decisions.md](docs/decisions.md) | All design decisions with rationale |
| [docs/inference.md](docs/inference.md) | Inference pipeline, latency, benchmarking |
| [docs/setup.md](docs/setup.md) | Training, API, Docker, testing, env vars |
| [docs/benchmarks.md](docs/benchmarks.md) | PyTorch vs ONNX comparison, per-stage latency |
| [docs/devpost.md](docs/devpost.md) | Hackathon write-up (problem, build, learnings) |
| [docs/README.md](docs/README.md) | Documentation index |

## Tech Stack

| Component | Choice |
| ----------- | -------- |
| SSL backbone | WavLM-large |
| Inference | ONNX Runtime |
| Classifier | MLP (1024->256->128->1) |
| Calibration | Isotonic regression |
| API | FastAPI + uvicorn |
| Web UI | Next.js 16 + React 19 + Tailwind 4 |
| Augmentation | Deterministic telephony variants (numpy, seed 2026) |
| Deployment | Docker multi-stage, GHCR images, compose dev/prod split |
| Package manager | uv (Python) + pnpm (web) |
