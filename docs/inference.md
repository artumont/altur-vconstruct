# Inference Pipeline

Step-by-step breakdown of what happens when `POST /detect` is called.

## Pipeline Steps

1. **Decode** — base64 string -> raw WAV bytes (`base64.b64decode`)
2. **Read** — WAV bytes -> stereo float32 array (`soundfile.read`)
3. **Extract channel** — take column 0 (caller) from stereo array -> mono tensor
4. **Resample** — 8 kHz -> 16 kHz via `torchaudio.transforms.Resample`
5. **Window** — chop into 3s chunks (48000 samples) with 50% overlap (hop = 24000)
6. **Sample** — select highest-energy caller window (`max_windows=1`)
7. **Embed** — run ONNX WavLM-large -> mean-pool over time -> (1, 1024)
8. **Classify** — run window embedding through MLP -> synthetic score
9. **Calibrate** — isotonic regressor -> calibrated synthetic probability
10. **Recenter** — map `decision_threshold=0.15` to response boundary 0.5
11. **Threshold** — recentered probability >= 0.5 -> `is_synthetic` boolean

## Key Functions

| Step | Function | Location |
| ------ | ---------- | ---------- |
| WAV decode | `decode_base64_wav()` | `pipeline/service.py` |
| Read stereo | `sf.read(io.BytesIO(wav_bytes))` | `pipeline/service.py` |
| Resample | `Resampler.resample_tensor()` | `audio/resampler.py` |
| Window | `window_waveform()` | `audio/windowing.py` |
| Sample windows | `sample_windows()` | `pipeline/service.py` |
| Embed | `ONNXSSLEvaluator.extract_batch()` | `models/extractor.py` |
| Classify | `SpoofClassifier.forward()` | `models/classifier.py` |
| Calibrate | `IsotonicRegression.predict()` | loaded in `pipeline/bundle.py` |

## Latency Profile

The dominant cost is WavLM extraction (step 7). Everything else is negligible.

With `max_windows=1`, every non-empty call performs one WavLM extraction.
The selected window has highest caller-channel energy, avoiding silence while
keeping encoder cost independent of call duration. ONNX Runtime uses four
intra-op threads and disables idle thread spinning.

Latency remains hardware-dependent. On an AMD Ryzen 5 PRO 4650U running on
battery, a sustained 213-call augmented-validation run measured 0.999s mean,
0.795s median, and 1.637s p95 HTTP latency. Internal processing averaged
0.957s; thermal throttling caused five calls to exceed 2s.

## Benchmarking

Run the benchmark suite:

```bash
cd apps/api
uv run python -m tests.benchmark.onnx_comparison
```

Tests extraction and full-pipeline latency across 2s, 5s, 10s, and 20s audio clips. Reports mean, median, p95 timing.

### What it measures

- **Extract**: ONNX WavLM embedding extraction only (the bottleneck)
- **Full pipeline**: decode -> resample -> window -> embed -> classify (end-to-end)

### Interpreting results

- Extract time is capped at one selected window
- Full pipeline adds ~5-10ms for resampling + classification
- p95 matters more than mean for judge scoring (worst-case latency)
