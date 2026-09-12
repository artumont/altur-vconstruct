# Inference Pipeline

Step-by-step breakdown of what happens when `POST /detect` is called.

## Pipeline Steps

1. **Decode** — base64 string -> raw WAV bytes (`base64.b64decode`)
2. **Read** — WAV bytes -> stereo float32 array (`soundfile.read`)
3. **Extract channel** — take column 0 (caller) from stereo array -> mono tensor
4. **Resample** — 8 kHz -> 16 kHz via `torchaudio.transforms.Resample`
5. **Window** — chop into 4s chunks (64000 samples) with 50% overlap (hop = 32000)
6. **Sample** — cap at `max_windows=2` evenly-spaced windows (`np.linspace`)
7. **Embed** — batch through ONNX WavLM-large -> mean-pool over time -> (B, 1024)
8. **Classify** — each window through MLP -> (B,) probabilities
9. **Aggregate** — mean across windows -> single score
10. **Calibrate** — isotonic regressor -> final confidence
11. **Threshold** — confidence >= 0.5 -> `is_synthetic` boolean

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

With `max_windows=2`:

- 2s audio: 1 window -> ~1 extraction
- 5s audio: 2 windows -> ~2 extractions
- 20s audio: 2 windows -> ~2 extractions (capped)
- 180s audio: 2 windows -> ~2 extractions (capped)

ONNX Runtime with 6 intra-op threads handles each window in ~100-200ms on CPU.

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

- Extract time scales with window count (capped at 2)
- Full pipeline adds ~5-10ms for resampling + classification
- p95 matters more than mean for judge scoring (worst-case latency)
