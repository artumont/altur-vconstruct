"""ONNX Runtime inference benchmark — per-stage pipeline timing.

Run:
    source apps/api/.venv/bin/activate
    python -m tests.benchmark.onnx_comparison
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

# ── project imports ─────────────────────────────────────────────────────────
_API_SRC = Path(__file__).resolve().parents[2] / "apps" / "api" / "src"
if str(_API_SRC) not in sys.path:
    sys.path.insert(0, str(_API_SRC))

from models.extractor import ONNXSSLEvaluator  # pyright: ignore[reportMissingImports]
from models.classifier import SpoofClassifier  # pyright: ignore[reportMissingImports]
from audio.resampler import Resampler  # pyright: ignore[reportMissingImports]
from pipeline.service import sample_windows  # pyright: ignore[reportMissingImports]
from audio.windowing import window_waveform  # pyright: ignore[reportMissingImports]
from models.calibrator import load_calibrator  # pyright: ignore[reportMissingImports]

# ── config ──────────────────────────────────────────────────────────────────
WARMUP_RUNS = 3
BENCH_RUNS = 15
AUDIO_DURATIONS_S = [2, 5, 10, 20]
SAMPLE_RATE = 8000
MAX_WINDOWS = 2


def make_stereo_wav_bytes(duration_s: float, sr: int = SAMPLE_RATE) -> bytes:
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


def bench_extract(ssl: ONNXSSLEvaluator, windows: list[torch.Tensor], n_runs: int) -> list[float]:
    batch = torch.stack(windows)
    times = []
    for _ in range(n_runs):
        gc.disable()
        t0 = time.perf_counter()
        ssl.extract_batch(batch)
        times.append(time.perf_counter() - t0)
        gc.enable()
        gc.collect()
    return times


def bench_full_pipeline(
    ssl: ONNXSSLEvaluator,
    resampler: Resampler,
    classifier: SpoofClassifier,
    wav_bytes: bytes,
    n_runs: int,
) -> list[float]:
    import soundfile as sf
    times = []
    for _ in range(n_runs):
        gc.disable()
        t0 = time.perf_counter()
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        caller = torch.from_numpy(np.ascontiguousarray(data[:, 0]))
        waveform = resampler.resample_tensor(caller, source_sr=sr)
        windows = window_waveform(waveform)
        windows = sample_windows(windows, MAX_WINDOWS)
        batch = torch.stack(windows)
        embs = ssl.extract_batch(batch).float()
        classifier(embs).squeeze(1).cpu().numpy()
        times.append(time.perf_counter() - t0)
        gc.enable()
        gc.collect()
    return times


def stats(times: list[float], skip: int = 0) -> dict[str, float]:
    ms = sorted(t * 1000 for t in times[skip:])
    n = len(ms)
    return {
        "mean": statistics.mean(ms),
        "median": statistics.median(ms),
        "p95": ms[int(n * 0.95)] if n >= 2 else ms[-1],
        "min": min(ms),
        "max": max(ms),
    }


def main():
    print("=" * 80)
    print("  ONNX Runtime Inference Benchmark")
    print("=" * 80)
    print(f"  Warmup: {WARMUP_RUNS}  |  Bench: {BENCH_RUNS}  |  Max windows: {MAX_WINDOWS}")

    print("\nLoading models...", flush=True)
    t0 = time.perf_counter()
    ssl = ONNXSSLEvaluator(device="cpu", intra_threads=6)
    print(f"  ✓ ONNX loaded in {time.perf_counter() - t0:.1f}s")

    from config import get_settings  # pyright: ignore[reportMissingImports]
    settings = get_settings()
    clf = SpoofClassifier(input_dim=1024)
    state = torch.load(settings.resolved_model_path, weights_only=True, map_location="cpu")
    clf.load_state_dict(state)
    clf.eval()
    resampler = Resampler(source_sr=8000, target_sr=16000)
    print("  ✓ Ready\n")

    for dur in AUDIO_DURATIONS_S:
        wav = make_stereo_wav_bytes(dur)
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(wav), dtype="float32")
        caller = torch.from_numpy(np.ascontiguousarray(data[:, 0]))
        waveform = resampler.resample_tensor(caller, source_sr=sr)
        windows = window_waveform(waveform)
        windows = sample_windows(windows, MAX_WINDOWS)
        n_win = len(windows)

        print(f"── {dur}s audio ({n_win} windows) ──")

        times_ext = bench_extract(ssl, windows, WARMUP_RUNS + BENCH_RUNS)
        times_full = bench_full_pipeline(ssl, resampler, clf, wav, WARMUP_RUNS + BENCH_RUNS)
        s_ext = stats(times_ext, skip=WARMUP_RUNS)
        s_full = stats(times_full, skip=WARMUP_RUNS)

        print(f"  Extract:    {s_ext['mean']:>6.0f}ms (p95={s_ext['p95']:.0f}ms)")
        print(f"  Full pipe:  {s_full['mean']:>6.0f}ms (p95={s_full['p95']:.0f}ms)")
        print()


if __name__ == "__main__":
    main()
