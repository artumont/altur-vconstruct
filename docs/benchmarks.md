# Benchmarks

Performance measurements for the voice anti-spoofing pipeline.

## PyTorch vs ONNX Runtime

Head-to-head comparison of WavLM-large embedding extraction.

### How to Run

```bash
cd apps/api
uv run python -m tests.benchmark.pytorch_vs_onnx
```

Requires both `torch` + `transformers` (PyTorch path) and `onnxruntime` (ONNX path). Downloads WavLM-large (~1.2GB) on first run.

### What It Measures

- **PyTorch**: `WavLMModel.from_pretrained()` via HuggingFace transformers, CPU inference
- **ONNX**: `onnxruntime.InferenceSession` with graph optimization, 6 intra-op threads
- Both extract 1024-dim embeddings from 4s audio windows (mean-pooled over time)
- Identical input: same synthetic WAV windows, same number of windows per duration

### Expected Results

ONNX Runtime consistently outperforms PyTorch for single-batch CPU inference due to:

- Graph-level optimizations (operator fusion, constant folding)
- No Python GIL overhead in the inference hot path
- Optimized memory layout for CPU execution

| Duration | Windows | PyTorch (mean) | ONNX (mean) | Speedup |
|----------|---------|----------------|-------------|---------|
| 2s       | 1       | ~180ms         | ~90ms       | ~2.0x   |
| 5s       | 2       | ~350ms         | ~170ms      | ~2.1x   |
| 10s      | 2       | ~350ms         | ~170ms      | ~2.1x   |
| 20s      | 2       | ~350ms         | ~170ms      | ~2.1x   |

*Results vary by CPU. Numbers above are representative for a 12-core machine.*

Key observations:

- ONNX load time is faster (~2s vs ~5s for PyTorch)
- Speedup is consistent across durations (2.0-2.2x)
- With `max_windows=2`, longer audio doesn't increase extraction time (capped)
- PyTorch benefits more from batching; ONNX wins on single-batch latency

### Why This Matters

The challenge scores **latency**. A 3-minute call with all windows (~45) would take:

- PyTorch: ~8s extraction
- ONNX: ~4s extraction

With window sampling (max_windows=2), both drop to ~2 extractions. But ONNX still wins on per-extraction time, which compounds across the judge's test set.

## Per-Stage Pipeline Breakdown

Detailed timing for each pipeline stage.

### How to Run

```bash
cd apps/api
uv run python -m tests.benchmark.inference_latency
```

### Pipeline Stages

| Stage | Operation | Expected Time |
| ------- | ----------- | --------------- |
| 1_decode_wav | base64 decode + soundfile read | <1ms |
| 2_resample | 8kHz -> 16kHz via torchaudio | <1ms |
| 3_window | chop into 4s chunks | <1ms |
| 4_sample_windows | cap to max_windows | <0.1ms |
| 5_wavlm_extract | ONNX WavLM embedding | ~85-170ms |
| 6_mlp_classify | 3-layer MLP forward | <1ms |
| 7_calibrate | isotonic regression predict | <0.1ms |

**Bottleneck**: Stage 5 (WavLM extraction) accounts for >95% of pipeline time.

### End-to-End Latency

With `max_windows=2` and ONNX Runtime:

| Audio Duration | E2E Latency (p95) | Throughput |
|----------------|-------------------|------------|
| 2s             | ~100ms            | ~10 req/s  |
| 5s             | ~180ms            | ~5 req/s   |
| 10s            | ~180ms            | ~5 req/s   |
| 20s            | ~180ms            | ~5 req/s   |

Throughput is single-concurrent. With FastAPI async + multiple workers, effective throughput scales linearly.

## ONNX Comparison (Single Backend)

Per-pipeline timing using only the ONNX backend (production configuration).

### How to Run

```bash
cd apps/api
uv run python -m tests.benchmark.onnx_comparison
```

Reports extraction-only and full-pipeline timing across audio durations. Useful for profiling the ONNX path in isolation.

## Configuration

### Benchmark Parameters

| Parameter | Value | Notes |
| ----------- | ------- | ------- |
| Warmup runs | 3 | Discarded from stats |
| Bench runs | 15 | Measured iterations |
| Audio durations | 2s, 5s, 10s, 20s | Simulates short to long calls |
| Input sample rate | 8 kHz | Telephony standard |
| Target sample rate | 16 kHz | WavLM requirement |
| Max windows | 2 | Latency optimization |
| ONNX threads | 6 | intra_op_num_threads |

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
