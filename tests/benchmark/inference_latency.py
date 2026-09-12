"""Inference latency benchmark — times every pipeline stage.

Pipeline: WAV bytes → resample → window → WavLM embed → MLP classify → calibrate

Run:
    source apps/api/.venv/bin/activate
    python -m tests.benchmark.inference_latency

Results print as a table + percentiles. No pytest required.
"""

from __future__ import annotations

import gc
import io
import statistics
import struct
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

# ── project imports (from apps/api/src) ────────────────────────────────────
_API_SRC = Path(__file__).resolve().parents[2] / "apps" / "api" / "src"
if str(_API_SRC) not in sys.path:
    sys.path.insert(0, str(_API_SRC))

from pipeline.bundle import ModelBundle, bundle  # pyright: ignore[reportMissingImports]
from config import get_settings  # pyright: ignore[reportMissingImports]
from models.extractor import ONNXSSLEvaluator  # pyright: ignore[reportMissingImports]
from models.classifier import SpoofClassifier  # pyright: ignore[reportMissingImports]
from audio.resampler import Resampler  # pyright: ignore[reportMissingImports]
from pipeline.service import sample_windows  # pyright: ignore[reportMissingImports]
from audio.windowing import window_waveform  # pyright: ignore[reportMissingImports]


# ── config ──────────────────────────────────────────────────────────────────
WARMUP_RUNS = 2          # first N iterations discarded from stats
BENCH_RUNS = 10          # measured iterations
AUDIO_DURATIONS_S = [2, 5, 10, 20]  # simulate short → long calls
SAMPLE_RATE = 8000       # telephony input
TARGET_SR = 16000        # WavLM expects 16 kHz
MAX_WINDOWS = 4          # matches settings default


# ── synthetic WAV generator ─────────────────────────────────────────────────
def make_stereo_wav_bytes(duration_s: float, sr: int = SAMPLE_RATE) -> bytes:
    """Generate random stereo 16-bit PCM WAV bytes (deterministic)."""
    n_frames = int(sr * duration_s)
    rng = np.random.default_rng(seed=42)
    samples = rng.integers(-32768, 32767, size=(n_frames, 2), dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


# ── timing helper ───────────────────────────────────────────────────────────
class StageTimer:
    """Context manager that accumulates wall-clock times."""

    def __init__(self):
        self.times: dict[str, list[float]] = {}

    def record(self, name: str, elapsed: float):
        self.times.setdefault(name, []).append(elapsed)

    def clear(self):
        self.times.clear()

    def report(self) -> dict[str, dict[str, float]]:
        """Return {stage: {mean, median, p95, p99, min, max}} in ms."""
        out: dict[str, dict[str, float]] = {}
        for name, vals in self.times.items():
            ms = sorted(v * 1000 for v in vals)
            n = len(ms)
            out[name] = {
                "mean": statistics.mean(ms),
                "median": statistics.median(ms),
                "min": min(ms),
                "max": max(ms),
                "p95": ms[int(n * 0.95)] if n >= 2 else ms[-1],
                "p99": ms[int(n * 0.99)] if n >= 2 else ms[-1],
                "n": n,
            }
        return out


# ── per-stage timing ────────────────────────────────────────────────────────
def benchmark_stages(
    resampler: Resampler,
    ssl: ONNXSSLEvaluator,
    classifier: SpoofClassifier,
    calibrator,
    wav_bytes: bytes,
    timer: StageTimer,
    max_windows: int = MAX_WINDOWS,
):
    """Run the full pipeline, recording time for each stage."""
    import soundfile as sf  # noqa: E402

    device = next(classifier.parameters()).device

    # Stage 1: decode WAV bytes
    t0 = time.perf_counter()
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    caller = torch.from_numpy(np.ascontiguousarray(data[:, 0]))
    timer.record("1_decode_wav", time.perf_counter() - t0)

    # Stage 2: resample 8kHz → 16kHz
    t0 = time.perf_counter()
    waveform = resampler.resample_tensor(caller, source_sr=sr)
    timer.record("2_resample", time.perf_counter() - t0)

    # Stage 3: window into 4s chunks
    t0 = time.perf_counter()
    windows = window_waveform(waveform)
    timer.record("3_window", time.perf_counter() - t0)

    # Stage 4: sample / cap windows
    t0 = time.perf_counter()
    windows = sample_windows(windows, max_windows)
    timer.record("4_sample_windows", time.perf_counter() - t0)

    # Stage 5: WavLM embedding extraction (THE bottleneck)
    t0 = time.perf_counter()
    with torch.no_grad():
        batch = torch.stack(windows).to(dtype=torch.float32, device=device)
        embs = ssl.extract_batch(batch).float().cpu()
    timer.record("5_wavlm_extract", time.perf_counter() - t0)

    # Stage 6: MLP classifier
    t0 = time.perf_counter()
    with torch.no_grad():
        scores = classifier(embs.to(device)).squeeze(1).cpu().numpy()
    timer.record("6_mlp_classify", time.perf_counter() - t0)

    # Stage 7: calibrator
    t0 = time.perf_counter()
    mean_score = float(scores.mean())
    calibrator.predict([mean_score])
    timer.record("7_calibrate", time.perf_counter() - t0)


# ── end-to-end timing ──────────────────────────────────────────────────────
def benchmark_e2e(bundle: ModelBundle, wav_bytes: bytes, timer: StageTimer):
    """Time the full classify_wav_bytes as a black box."""
    from service import classify_wav_bytes  # pyright: ignore[reportMissingImports]

    t0 = time.perf_counter()
    classify_wav_bytes(wav_bytes)
    timer.record("E2E_full", time.perf_counter() - t0)


# ── pretty print ────────────────────────────────────────────────────────────
def print_report(reports: dict[str, dict[str, float]], title: str):
    print(f"\n{'=' * 82}")
    print(f"  {title}")
    print(f"{'=' * 82}")
    hdr = f"{'Stage':<25} {'mean':>8} {'median':>8} {'p95':>8} {'p99':>8} {'min':>8} {'max':>8}"
    print(hdr)
    print("-" * 82)
    total_mean = 0.0
    for name, stats in reports.items():
        if name == "E2E_full":
            print("-" * 82)
        row = (
            f"{name:<25}"
            f" {stats['mean']:>7.1f}ms"
            f" {stats['median']:>7.1f}ms"
            f" {stats['p95']:>7.1f}ms"
            f" {stats['p99']:>7.1f}ms"
            f" {stats['min']:>7.1f}ms"
            f" {stats['max']:>7.1f}ms"
        )
        print(row)
        if name.startswith(("1_", "2_", "3_", "4_", "5_", "6_", "7_")):
            total_mean += stats["mean"]
    print("-" * 82)
    print(f"{'Stages sum':<25} {total_mean:>7.1f}ms")
    if "E2E_full" in reports:
        e2e = reports["E2E_full"]["mean"]
        print(f"{'E2E measured':<25} {e2e:>7.1f}ms")
        print(f"{'Overhead':<25} {e2e - total_mean:>7.1f}ms")
    print()


def print_bottleneck(reports: dict[str, dict[str, float]]):
    """Highlight the slowest stage."""
    stage_keys = [k for k in reports if k.startswith(("1_", "2_", "3_", "4_", "5_", "6_", "7_"))]
    if not stage_keys:
        return
    slowest = max(stage_keys, key=lambda k: reports[k]["mean"])
    total = sum(reports[k]["mean"] for k in stage_keys)
    pct = reports[slowest]["mean"] / total * 100 if total > 0 else 0
    print(f"  ⚠  BOTTLENECK: {slowest} = {reports[slowest]['mean']:.1f}ms ({pct:.0f}% of pipeline)")
    print()


# ── main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 82)
    print("  INFERENCE LATENCY BENCHMARK")
    print("  WavLM-large + SpoofClassifier + Isotonic Calibrator")
    print("=" * 82)

    settings = get_settings()
    print(f"  Device: {settings.device}  |  dtype: {settings.dtype}")
    print(f"  Warmup: {WARMUP_RUNS}  |  Bench: {BENCH_RUNS}")
    print(f"  Max windows: {MAX_WINDOWS}  |  Batch size: {settings.extract_batch_size}")

    # ── load model stack ────────────────────────────────────────────────
    print("\nLoading model stack...", flush=True)
    t_load = time.perf_counter()
    bundle = ModelBundle()
    bundle.ensure()
    print(f"  ✓ Loaded in {time.perf_counter() - t_load:.1f}s")

    assert bundle.ssl is not None
    assert bundle.classifier is not None
    assert bundle.calibrator is not None
    assert bundle.resampler is not None

    # ── benchmark per duration ──────────────────────────────────────────
    for dur in AUDIO_DURATIONS_S:
        wav = make_stereo_wav_bytes(dur)
        # approximate window count: 50% overlap, 4s windows
        n_samples_16k = dur * TARGET_SR
        n_windows_raw = max(1, int((n_samples_16k - 64000) / 32000) + 1)
        n_windows_capped = min(n_windows_raw, MAX_WINDOWS)
        print(f"\n── Audio: {dur}s  (~{n_windows_raw} raw windows → {n_windows_capped} scored) ──")

        timer = StageTimer()

        # warmup
        print(f"  warmup ({WARMUP_RUNS})...", end="", flush=True)
        for _ in range(WARMUP_RUNS):
            benchmark_stages(
                bundle.resampler, bundle.ssl, bundle.classifier,
                bundle.calibrator, wav, timer,
            )
        timer.clear()
        print(" done")

        # bench per-stage
        print(f"  bench ({BENCH_RUNS})...", end="", flush=True)
        for i in range(BENCH_RUNS):
            gc.disable()
            benchmark_stages(
                bundle.resampler, bundle.ssl, bundle.classifier,
                bundle.calibrator, wav, timer,
            )
            gc.enable()
            gc.collect()
            if (i + 1) % 5 == 0:
                print(".", end="", flush=True)
        print(" done")

        reports = timer.report()
        print_report(reports, f"Per-stage breakdown — {dur}s audio")
        print_bottleneck(reports)

        # bench E2E
        timer_e2e = StageTimer()
        print(f"  E2E bench ({BENCH_RUNS})...", end="", flush=True)
        for _ in range(WARMUP_RUNS):
            benchmark_e2e(bundle, wav, timer_e2e)
        timer_e2e.clear()
        for i in range(BENCH_RUNS):
            gc.disable()
            benchmark_e2e(bundle, wav, timer_e2e)
            gc.enable()
            gc.collect()
            if (i + 1) % 5 == 0:
                print(".", end="", flush=True)
        print(" done")

        e2e_report = timer_e2e.report()
        print_report(e2e_report, f"End-to-end — {dur}s audio")

        if "E2E_full" in e2e_report:
            e2e = e2e_report["E2E_full"]
            rps = 1000 / e2e["mean"] if e2e["mean"] > 0 else 0
            print(f"  Throughput: ~{rps:.2f} req/s (single concurrent)")
            print(f"  E2E p95: {e2e['p95']:.1f}ms  |  p99: {e2e['p99']:.1f}ms")

    # ── summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 82)
    print("  MODEL WEIGHTS SUMMARY")
    print("=" * 82)

    clf_params = sum(p.numel() for p in bundle.classifier.parameters())
    wavlm_params = sum(p.numel() for p in bundle.ssl.model.parameters())
    print(f"  WavLM-large params:    {wavlm_params:>12,}")
    print(f"  Classifier params:     {clf_params:>12,}")
    print(f"  Total model params:    {wavlm_params + clf_params:>12,}")
    print(f"  Model size (WavLM):    ~{wavlm_params * 4 / 1e9:.1f} GB (float32)")
    print(f"  Model size (clf):      ~{clf_params * 4 / 1e3:.0f} KB (float32)")
    print()


if __name__ == "__main__":
    main()
