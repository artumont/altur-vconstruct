"""PyTorch vs ONNX Runtime head-to-head benchmark.

Compares WavLM-large inference via PyTorch (SSLEvaluator) vs ONNX Runtime
(ONNXSSLEvaluator) across audio durations.

Run:
    cd apps/api
    uv run python -m tests.benchmark.pytorch_vs_onnx
"""

from __future__ import annotations

import gc
import io
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

# ── project imports ─────────────────────────────────────────────────────────
_API_SRC = Path(__file__).resolve().parents[2] / "apps" / "api" / "src"
_TRAIN_SRC = Path(__file__).resolve().parents[2] / "apps" / "train" / "src"
if str(_API_SRC) not in sys.path:
    sys.path.insert(0, str(_API_SRC))
if str(_TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(_TRAIN_SRC))

from models.extractor import ONNXSSLEvaluator  # pyright: ignore[reportMissingImports]

# ── config ──────────────────────────────────────────────────────────────────
WARMUP_RUNS = 3
BENCH_RUNS = 15
AUDIO_DURATIONS_S = [2, 5, 10, 20]
SAMPLE_RATE = 8000
TARGET_SR = 16000
MAX_WINDOWS = 1
WINDOW_SAMPLES = 48000  # 3s @ 16kHz
HOP = WINDOW_SAMPLES // 2


# ── helpers ─────────────────────────────────────────────────────────────────

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


def make_windows(duration_s: float) -> list[torch.Tensor]:
    """Create synthetic 3s windows at 16kHz."""
    n_samples_16k = int(duration_s * TARGET_SR)
    if n_samples_16k < WINDOW_SAMPLES:
        pad = torch.randn(WINDOW_SAMPLES)
        return [pad]
    windows = []
    start = 0
    while start + WINDOW_SAMPLES <= n_samples_16k:
        windows.append(torch.randn(WINDOW_SAMPLES))
        start += HOP
    # cap
    if len(windows) > MAX_WINDOWS:
        idx = np.linspace(0, len(windows) - 1, MAX_WINDOWS).round().astype(int)
        windows = [windows[i] for i in idx]
    return windows


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


def fmt(d: dict[str, float]) -> str:
    return f"mean={d['mean']:>7.1f}ms  med={d['median']:>7.1f}ms  p95={d['p95']:>7.1f}ms"


# ── PyTorch extractor (vendored from train) ────────────────────────────────

class PyTorchSSLEvaluator:
    """Frozen WavLM-large via PyTorch. Same interface as ONNXSSLEvaluator."""

    def __init__(self, device: str = "cpu"):
        from transformers import WavLMModel  # pyright: ignore[reportMissingImports]
        self.device = torch.device(device)
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(self.device)  # type: ignore[arg-type]
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        inputs = waveforms.to(self.device)
        outputs = self.model(inputs)
        return outputs.last_hidden_state.mean(dim=1)


# ── benchmark functions ────────────────────────────────────────────────────

def bench_pytorch(
    pt_eval: PyTorchSSLEvaluator, windows: list[torch.Tensor], n_runs: int
) -> list[float]:
    batch = torch.stack(windows)
    times = []
    for _ in range(n_runs):
        gc.disable()
        t0 = time.perf_counter()
        pt_eval.extract_batch(batch)
        times.append(time.perf_counter() - t0)
        gc.enable()
        gc.collect()
    return times


def bench_onnx(
    ort_eval: ONNXSSLEvaluator, windows: list[torch.Tensor], n_runs: int
) -> list[float]:
    batch = torch.stack(windows)
    times = []
    for _ in range(n_runs):
        gc.disable()
        t0 = time.perf_counter()
        ort_eval.extract_batch(batch)
        times.append(time.perf_counter() - t0)
        gc.enable()
        gc.collect()
    return times


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  PyTorch vs ONNX Runtime — WavLM-large Embedding Extraction")
    print("=" * 80)
    print(f"  Warmup: {WARMUP_RUNS}  |  Bench: {BENCH_RUNS}  |  Max windows: {MAX_WINDOWS}")
    print(f"  Device: CPU  |  Threads: 6 (ONNX intra_op)")
    print()

    # ── load ONNX ───────────────────────────────────────────────────────
    print("Loading ONNX model...", flush=True)
    t0 = time.perf_counter()
    ort_eval = ONNXSSLEvaluator(device="cpu", intra_threads=6)
    ort_load = time.perf_counter() - t0
    print(f"  ✓ ONNX loaded in {ort_load:.1f}s")

    # ── load PyTorch ────────────────────────────────────────────────────
    print("Loading PyTorch model...", flush=True)
    t0 = time.perf_counter()
    pt_eval = PyTorchSSLEvaluator(device="cpu")
    pt_load = time.perf_counter() - t0
    print(f"  ✓ PyTorch loaded in {pt_load:.1f}s")
    print()

    # ── warmup both ─────────────────────────────────────────────────────
    print("Warming up...", flush=True)
    warmup_windows = make_windows(5.0)
    for _ in range(WARMUP_RUNS):
        pt_eval.extract_batch(torch.stack(warmup_windows))
        ort_eval.extract_batch(torch.stack(warmup_windows))
    print("  ✓ Ready\n")

    # ── benchmark ───────────────────────────────────────────────────────
    results = []

    for dur in AUDIO_DURATIONS_S:
        windows = make_windows(dur)
        n_win = len(windows)
        print(f"── {dur}s audio ({n_win} windows) ──")

        pt_times = bench_pytorch(pt_eval, windows, WARMUP_RUNS + BENCH_RUNS)
        ort_times = bench_onnx(ort_eval, windows, WARMUP_RUNS + BENCH_RUNS)

        pt_s = stats(pt_times, skip=WARMUP_RUNS)
        ort_s = stats(ort_times, skip=WARMUP_RUNS)

        speedup = pt_s["mean"] / ort_s["mean"] if ort_s["mean"] > 0 else 0

        print(f"  PyTorch: {fmt(pt_s)}")
        print(f"  ONNX:    {fmt(ort_s)}")
        print(f"  Speedup: {speedup:.2f}x faster (ONNX)")
        print()

        results.append({
            "duration": dur,
            "windows": n_win,
            "pytorch": pt_s,
            "onnx": ort_s,
            "speedup": speedup,
        })

    # ── summary table ───────────────────────────────────────────────────
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  {'Duration':>10}  {'Windows':>7}  {'PyTorch':>12}  {'ONNX':>12}  {'Speedup':>8}")
    print("  " + "-" * 65)
    for r in results:
        print(
            f"  {r['duration']:>7}s  {r['windows']:>7}  "
            f"{r['pytorch']['mean']:>9.1f}ms  "
            f"{r['onnx']['mean']:>9.1f}ms  "
            f"{r['speedup']:>6.2f}x"
        )

    avg_speedup = statistics.mean(r["speedup"] for r in results)
    print("  " + "-" * 65)
    print(f"  {'Average':>10}  {'':>7}  {'':>12}  {'':>12}  {avg_speedup:>6.2f}x")
    print()

    # ── model info ──────────────────────────────────────────────────────
    print("  Model: microsoft/wavlm-large (~300M params)")
    print(f"  PyTorch load time: {pt_load:.1f}s")
    print(f"  ONNX load time:    {ort_load:.1f}s")
    print()


if __name__ == "__main__":
    main()
