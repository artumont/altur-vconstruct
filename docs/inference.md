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
12. **Confidence** — `max(probability, 1 - probability)` rounded to 4 decimals

The response separates the decision from its magnitude: `is_synthetic` comes from
the recentered probability crossing 0.5, while `confidence` is the distance to the
nearest decision (always >= 0.5), i.e. confidence in whichever class was returned.
A tie-break-heavy judge gets a number that is high only when the calibrated
probability is close to 0 or 1.

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

With `max_windows=1`, every non-empty call performs exactly one WavLM extraction.
The selected window has the highest caller-channel energy, which avoids spending
the only extraction on silence, and keeps encoder cost independent of call
duration. ONNX Runtime uses four intra-op threads and disables idle thread
spinning.

Measured with `tests/benchmark/inference_latency.py` on an AMD Ryzen 5 9600X
(6C/12T, CPU-only, 4 intra-op threads):

| Audio | Raw windows | Scored | E2E mean | E2E p95 | Throughput |
| ----- | ----------- | ------ | -------- | ------- | ---------- |
| 2 s | 1 | 1 | 174 ms | 190 ms | ~5.8 req/s |
| 5 s | 2 | 1 | 202 ms | 234 ms | ~5.0 req/s |
| 10 s | 5 | 1 | 240 ms | 283 ms | ~4.2 req/s |
| 20 s | 12 | 1 | 194 ms | 236 ms | ~5.2 req/s |

WavLM extraction is ~170 ms of that and consistently ~90% of the end-to-end
`/detect` latency budget (99% of in-process stage time, where transport is not
counted); every other stage is under 2 ms.
Throughput is single-concurrent — the model is CPU-bound, so more workers contend
for the same cores instead of scaling linearly.

Hardware state matters more than the weights. On deployment hardware `/detect`
averages **~0.5-0.6 s per call**; the earlier pipeline that scored every window
without thread tuning averaged **~1.0-1.2 s**, and a sustained 213-call HTTP run on
a throttled laptop on battery measured 0.999 s mean / 0.795 s median /
1.637 s p95. Re-run the benchmark on the deployment machine before quoting a
latency number.

## Benchmarking

Run the benchmark suite:

```bash
# from the repo root, with the API environment active (provides onnxruntime)
source apps/api/.venv/bin/activate
python -m tests.benchmark.onnx_comparison
python -m tests.benchmark.inference_latency
```

Tests extraction and full-pipeline latency across 2s, 5s, 10s, and 20s audio
clips (2 warmup runs, 10 measured). Reports mean, median, p95 and p99 timing.
Both scripts need `wavlm-large.onnx`, which the API image builds during
`docker build` and the repo does not commit — see
[setup.md](setup.md#benchmarks) for the export command.

### What it measures

- **Extract**: ONNX WavLM embedding extraction only (the bottleneck)
- **Full pipeline**: decode -> resample -> window -> embed -> classify (end-to-end)

### Interpreting results

- Extract time is capped at one selected window
- Full pipeline adds ~2-8ms on top of extraction for resampling, classification and calibration
- Deployed HTTP latency averages ~0.5-0.6 s per call, so transport and container overhead dominate the per-stage budget
- p95 matters more than mean for judge scoring (worst-case latency)
