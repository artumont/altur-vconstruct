# altur-vconstruct

**Tagline:** Reality has a voice. We know it.

## Inspiration

Voice deepfakes are reshaping fraud. A synthetic caller can impersonate a bank's own agent, a family member, or a government official, and the victim hears a perfectly human voice. Altur builds AI voice agents for banks across Latin America, handling millions of real calls. If synthetic voices infiltrate those calls, trust collapses.

We asked ourselves: can a model learn the subtle acoustic fingerprint that separates a real human vocal tract from a neural vocoder?

The answer is yes — and it can do it in about half a second per call on CPU.

## What it does

altur-vconstruct is a voice anti-spoofing API. Given a stereo phone call recording, it determines whether the incoming caller is a real human or a synthetic voice (TTS deepfake).

```
POST /detect
  Body:  {"audio_base64": "<stereo 8kHz WAV>"}
  Resp:  {"is_synthetic": false, "confidence": 0.92}
```

The system works on real telephony audio — 8 kHz, compressed, noisy — not clean lab recordings. It's designed to sit in-line with Altur's call infrastructure and score every call. A small Next.js UI ships with it for dragging in a WAV and seeing the verdict, the confidence and the endpoint's health.

## How we built it

**SSL + MLP two-stage architecture:**

1. **WavLM-large** (Microsoft, 300M params) as a frozen self-supervised backbone: it extracts 1024-dimensional acoustic embeddings from 4-second audio windows during training, and from a single 3-second window at inference. We chose WavLM over wav2vec2 and XLS-R because it achieves the lowest EER (2.29%) on ASVspoof 2021, the standard anti-spoofing benchmark.

2. **3-layer MLP classifier** (1024 → 256 → 128 → 1, ~296K params) with BatchNorm and Dropout — trained on frozen embeddings. With only 282 training calls, we couldn't fine-tune the backbone without overfitting. The MLP trains in seconds, enabling rapid iteration.

3. **Telephony augmentation of the training set** — our `apps/dataset` module generates deterministic variants of every training call (caller/agent latency shift, gain + 18-30 dB noise, packet-loss dropouts with clock-rate drift). That takes the training pool from 282 calls to **846 augmented calls (~800+ training calls)**, so accuracy holds up no matter which condition the judge's audio carries. Validation audio is never augmented.

4. **Isotonic calibration** — maps raw classifier scores to well-calibrated probabilities. The challenge uses `confidence` for tie-breaking, so arbitrary sigmoid outputs weren't acceptable.

5. **ONNX Runtime** — we export WavLM to ONNX at Docker build time, cutting inference latency ~2x on CPU versus PyTorch. The Docker image is lean: no transformers, no GPU libs.

**Key insight:** Instead of scoring all ~119 overlapping windows from a 3-minute call, we score the single highest-energy caller window. A human sounds human everywhere; a synthetic voice sounds synthetic everywhere. Picking the window by caller-channel energy keeps that one extraction away from silence or greetings.

## Challenges we ran into

- **Small dataset, big model.** 282 training calls with a 300M-parameter backbone. Fine-tuning was a trap, the model memorized speakers instead of learning spoof patterns. Frozen embeddings + lightweight head was the answer.

- **Too little clean audio.** The dataset is small and comparatively clean compared with real phone traffic. Rather than hope the judge sends the same conditions, we built a deterministic augmentation module and grew the training pool to ~800+ calls with noise, gain, packet-loss and latency-shift variants.

- **8 kHz telephony audio.** WavLM expects 16 kHz. The 8 kHz → 16 kHz resampling step is mandatory and adds a dependency (torchaudio). Getting the resampling right without introducing artifacts was non-trivial.

- **Latency is a scoring criterion.** Every design decision had to pass a latency budget. Window sampling, ONNX export, lazy model loading, all driven by the constraint that judges would time the endpoint.

- **Confidence calibration.** Raw sigmoid outputs are not probabilities. Platt scaling assumed a logistic shape that didn't match our bimodal score distribution. Isotonic regression solved it, but fitting it correctly required careful validation-set handling.

- **Docker image size.** Full PyTorch + transformers is ~2GB. The multi-stage build (download + ONNX export in builder, copy only ONNX model to runtime) cutting the image to a fraction of that.

## Accomplishments that we're proud of

- **~0.5-0.6s average inference per call** on CPU, down from ~1.0-1.2s before window selection and ONNX thread tuning, with well-calibrated confidence scores
- **Robust by data, not by size** — ~800+ augmented training calls from a 282-call split, so accuracy survives noise, level differences, packet loss and latency shifts
- **Clean architecture** — frozen SSL backbone, cached embeddings, separate train/API modules, ONNX runtime path
- **Thorough benchmarking** — PyTorch vs ONNX comparison, per-stage pipeline breakdown, latency profiling across audio durations
- **Production-ready deployment** — Docker multi-stage build, GHCR images, health endpoint, web UI, env-driven config, integration tests, plus a judge-style harness that scores the live endpoint the way the judges will
- **Honest documentation** — every design decision has a rationale, alternatives considered, and trade-offs acknowledged

## What we learned

- **Frozen SSL embeddings are surprisingly powerful.** You don't always need to fine-tune. With the right backbone and a small head, you can get strong results on tiny datasets.
- **Latency constraints shape architecture.** The single high-energy window trick was born from a scoring rule, not from ML theory, but it works because spoof signals are consistent across time.
- **Calibration matters more than accuracy.** A well-calibrated 90% confidence is more useful than a poorly-calibrated 99%. Judges notice.
- **ONNX is worth the export hassle.** 2x CPU speedup with zero accuracy loss is a free lunch for deployment.
- **Speaker-disjoint splits are non-negotiable.** Without them, the model learns "which speaker" not "human vs machine."

## What's next for altur-vconstruct

- **Conversational features** — use both channels (caller + agent) to detect response latency patterns, interruption behavior, and silence handling. The turn JSONs are already there.
- **Multi-engine robustness** — test against more TTS engines (Bark, XTTS, Parler, F5-TTS) to ensure the model generalizes beyond what Altur provided.
- **Streaming detection** — score incrementally as the call progresses, returning a verdict as early as possible instead of waiting for the full recording.
- **Model compression** — explore distillation or quantization to shrink WavLM for edge deployment.
- **Real-time integration** — plug into Altur's call pipeline as a middleware filter, flagging suspicious calls before they reach human agents.
