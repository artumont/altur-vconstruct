# Benchmarks

Performance measurements for the voice anti-spoofing pipeline.

## PyTorch vs ONNX Runtime

Head-to-head comparison of WavLM-large embedding extraction.

### How to Run

```bash
# from the repo root, with an environment that has both backends:
#   transformers + torch  -> apps/train
#   onnxruntime           -> apps/api
uv run --project apps/train --with onnxruntime python -m tests.benchmark.pytorch_vs_onnx
```

Requires `wavlm-large.onnx` (built by the API image, not committed) — see
[setup.md](setup.md#benchmarks) for the export command.

### What It Measures

- **PyTorch**: `WavLMModel.from_pretrained()` via HuggingFace transformers, CPU inference
- **ONNX**: `onnxruntime.InferenceSession` with graph optimization, 6 intra-op threads (the script pins 6 explicitly; production uses 4)
- Both extract 1024-dim embeddings from 3s inference windows (mean-pooled over time)
- Identical input: same synthetic WAV windows, same number of windows per duration

### Expected Results

ONNX Runtime consistently outperforms PyTorch for single-batch CPU inference due to:

- Graph-level optimizations (operator fusion, constant folding)
- No Python GIL overhead in the inference hot path
- Optimized memory layout for CPU execution

Development measurements showed a consistent **2.0-2.2x per-window speedup** on
CPU, with identical mean-pooled 1024-dim embeddings either way. Absolute latency
is hardware-dependent, so run the script on the deployment machine instead of
copying numbers; the ONNX side of the production pipeline is measured below.

Key observations:

- With `max_windows=1`, longer audio does not increase extraction time (capped)
- PyTorch benefits more from batching; ONNX wins on single-batch latency
- ONNX session creation is fast (~1-3 s) and needs no `transformers` at runtime

### Why This Matters

The challenge scores **latency**. A 3-minute call yields ~119 overlapping
3-second windows at 50% hop. At the ~170 ms per window measured below, scoring all
of them is roughly 20 s of encoder work for one call.

With `max_windows=1`, that drops to a single highest-energy caller window, i.e.
~99% less encoder work. The selected window avoids silence so the only extraction
is never wasted, and pipeline cost becomes independent of call length.

## Per-Stage Pipeline Breakdown

Detailed timing for each pipeline stage.

### How to Run

```bash
# from the repo root, with the API environment active (provides onnxruntime)
source apps/api/.venv/bin/activate
python -m tests.benchmark.inference_latency
```

### Pipeline Stages

| Stage | Operation | Measured (mean) |
| ------- | ----------- | --------------- |
| 1_decode_wav | base64 decode + soundfile read | 0.3-0.6ms |
| 2_resample | 8kHz -> 16kHz via torchaudio | 0.3-1.5ms |
| 3_window | chop into 3s chunks | <0.1ms |
| 4_sample_windows | select highest-energy window | 0.0-0.2ms |
| 5_wavlm_extract | ONNX WavLM embedding | 165-173ms |
| 6_mlp_classify | 3-layer MLP forward | ~0.3ms |
| 7_calibrate | isotonic regression predict | ~0.3ms |

**Bottleneck**: Stage 5 (WavLM extraction) is always ~90% of end-to-end `/detect`
latency (99% of in-process stage time, where transport and container overhead are
not counted). Optimizing anything else is noise until the encoder gets cheaper.

### End-to-End Latency

Measured on an AMD Ryzen 5 9600X (6C/12T, CPU-only, 4 ONNX intra-op threads,
2 warmup + 10 measured runs per duration):

| Audio | Raw windows | Scored | E2E mean | E2E median | E2E p95 | Throughput |
| ----- | ----------- | ------ | -------- | ---------- | ------- | ---------- |
| 2s | 1 | 1 | 174ms | 173ms | 190ms | ~5.8 req/s |
| 5s | 2 | 1 | 202ms | 202ms | 234ms | ~5.0 req/s |
| 10s | 5 | 1 | 240ms | 234ms | 283ms | ~4.2 req/s |
| 20s | 12 | 1 | 194ms | 193ms | 236ms | ~5.2 req/s |

Throughput is single-concurrent. The model is CPU-bound, so additional workers
contend for the same cores instead of scaling linearly.

These are in-process stage numbers on a fast desktop CPU. Against the deployed
endpoint the same one-window pipeline averages **~0.5-0.6 s per call** (the earlier
all-window pipeline without thread tuning averaged **~1.0-1.2 s**), with container
and transport overhead plus sustained thermal load accounting for the difference.
Run the benchmark on the deployment machine before quoting a latency number; do not
extrapolate from a dev box.

## ONNX Comparison (Single Backend)

Per-pipeline timing using only the ONNX backend, with the higher thread count the
comparison scripts pin.

### How to Run

```bash
# from the repo root, with the API environment active (provides onnxruntime)
source apps/api/.venv/bin/activate
python -m tests.benchmark.onnx_comparison
```

Reports extraction-only and full-pipeline timing across audio durations (3 warmup +
15 measured runs per duration). On the same 6-core machine: extraction 165-173 ms
mean (p95 175-191 ms) and full pipeline 169-173 ms mean (p95 185-202 ms) for 2-20 s
calls — within noise of the per-stage numbers above, confirming that nothing
outside WavLM extraction influences latency.

## Configuration

### Benchmark Parameters

| Parameter | Value | Notes |
| ----------- | ------- | ------- |
| Warmup runs | 3 | Discarded from stats |
| Bench runs | 15 | Measured iterations |
| Audio durations | 2s, 5s, 10s, 20s | Simulates short to long calls |
| Input sample rate | 8 kHz | Telephony standard |
| Target sample rate | 16 kHz | WavLM requirement |
| Max windows | 1 | Highest-energy caller window |
| ONNX threads | 4 | `inference_latency.py` reads the production config; the two comparison scripts pin 6 |

### Interpreting Results

- **mean**: Average latency — good for throughput estimates
- **median**: Midpoint — less affected by outliers
- **p95**: 95th percentile — what judges will mostly see
- **p99**: 99th percentile — worst-case spikes

For the challenge, **p95** is the most relevant metric. Judges run many calls; most will hit the p95 latency.

### Hardware Impact

Results vary significantly by CPU:

- Apple M-series: ONNX speedup ~1.8-2.2x (ARM optimization)
- Intel/AMD x86: ONNX speedup ~2.0-2.5x (AVX-512, VNNI)
- GPU: PyTorch wins (CUDA kernels), but challenge endpoint is CPU-only

The benchmark assumes CPU inference since the Docker deployment targets CPU.
