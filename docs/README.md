# altur-vconstruct — Docs

Detailed documentation for the voice anti-spoofing system.

## Documents

| Document | Contents |
| ---------- | ---------- |
| [architecture.md](architecture.md) | Model design, SSL backbone, MLP head, calibration, channel separation |
| [decisions.md](decisions.md) | All design decisions with rationale (window sampling, ONNX, calibration, etc.) |
| [inference.md](inference.md) | Step-by-step inference pipeline, latency profile, benchmarking |
| [setup.md](setup.md) | Training, API, Docker, testing, env vars, troubleshooting |
| [benchmarks.md](benchmarks.md) | PyTorch vs ONNX comparison, per-stage latency, benchmark scripts |
| [devpost.md](devpost.md) | Hackathon write-up: problem, build, learnings |

## Quick Links

- **Root README**: [../README.md](../README.md) — project overview and measured results
- **Challenge repo**: altur-vconstruct
- **Environment variables**: [setup.md](setup.md#environment-variables)
- **Shipped model metrics**: [../README.md](../README.md#measured-results)
