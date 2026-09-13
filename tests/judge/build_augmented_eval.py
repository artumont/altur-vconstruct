"""Create judge-compatible perturbations of original Altur calls.

Uses original validation calls only. Labels stay unchanged. Variants preserve
8 kHz stereo PCM16 format while introducing caller latency, gain/noise,
packet-loss-like dropouts, and small speed changes.

Usage:
    uv run --project tests python tests/judge/build_augmented_eval.py \
        --manifest apps/train/data/manifest.csv \
        --audio-dir /path/to/apps/train/audio \
        --out tests/judge/augmented_val \
        --variants 3
"""

import argparse
import csv
import hashlib
import random
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 8000


def read_wav(path: Path) -> np.ndarray:
    """Read stereo PCM16 WAV as frames x channels."""
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 2 or source.getframerate() != SAMPLE_RATE:
            raise ValueError(f"Expected stereo 8 kHz WAV: {path}")
        if source.getsampwidth() != 2:
            raise ValueError(f"Expected PCM16 WAV: {path}")
        frames = source.readframes(source.getnframes())
    return np.frombuffer(frames, dtype="<i2").reshape(-1, 2).copy()


def write_wav(path: Path, frames: np.ndarray) -> float:
    """Write stereo PCM16 WAV and return duration."""
    frames = np.clip(frames, -32768, 32767).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(frames.tobytes())
    return len(frames) / SAMPLE_RATE


def respeed(frames: np.ndarray, factor: float) -> np.ndarray:
    """Resample both channels by a small speed factor."""
    target_len = max(1, round(len(frames) / factor))
    source_x = np.linspace(0.0, 1.0, len(frames), endpoint=False)
    target_x = np.linspace(0.0, 1.0, target_len, endpoint=False)
    channels = [
        np.interp(target_x, source_x, frames[:, channel].astype(np.float32))
        for channel in range(2)
    ]
    return np.column_stack(channels)


def variant_latency(frames: np.ndarray, rng: random.Random) -> np.ndarray:
    """Delay caller channel while agent starts at original time."""
    delay_ms = rng.uniform(100.0, 1500.0)
    delay = round(delay_ms * SAMPLE_RATE / 1000.0)
    caller = np.pad(frames[:, 0], (delay, 0))
    agent = np.pad(frames[:, 1], (0, delay))
    return np.column_stack((caller, agent))


def variant_noise_gain(frames: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply call-level gain and additive telephony-like background noise."""
    gain_db = rng.uniform(-5.0, 5.0)
    gain = 10 ** (gain_db / 20.0)
    output = frames.astype(np.float32) * gain
    caller_rms = float(np.sqrt(np.mean(output[:, 0] ** 2)))
    snr_db = rng.uniform(18.0, 30.0)
    noise_rms = max(8.0, caller_rms / (10 ** (snr_db / 20.0)))
    noise = rng.normalvariate(0.0, noise_rms)
    # Use independent noise per frame; caller dominates at stated SNR.
    noise_array = np.random.default_rng(rng.randrange(2**32)).normal(
        0.0, abs(noise), size=output.shape
    )
    return output + noise_array


def variant_dropout_speed(frames: np.ndarray, rng: random.Random) -> np.ndarray:
    """Add short packet-loss gaps, then apply small clock-rate variation."""
    output = frames.astype(np.float32).copy()
    for _ in range(rng.randint(2, 8)):
        start = rng.randrange(len(output))
        length = min(rng.randint(50, 300) * SAMPLE_RATE // 1000, len(output) - start)
        output[start : start + length, :] = 0.0
    output = respeed(output, rng.uniform(0.97, 1.03))
    return output


def make_variant(frames: np.ndarray, kind: int, rng: random.Random) -> np.ndarray:
    """Apply one deterministic perturbation family."""
    if kind == 0:
        return variant_latency(frames, rng)
    if kind == 1:
        return variant_noise_gain(frames, rng)
    return variant_dropout_speed(frames, rng)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--out", default="tests/judge/augmented_val")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--variants", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    audio_dir = Path(args.audio_dir)
    out = Path(args.out)
    output_audio = out / "audio"
    output_audio.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(newline="", encoding="utf-8") as source:
        source_rows = [row for row in csv.DictReader(source) if row["split"] == args.split]
    if not source_rows:
        raise SystemExit(f"No rows for split {args.split!r}")

    rows = []
    for source_row in source_rows:
        source_id = source_row["anon_id"]
        source_path = audio_dir / f"{source_id}.wav"
        frames = read_wav(source_path)
        for kind in range(args.variants):
            seed_material = f"{args.seed}:{source_id}:{kind}".encode()
            digest = hashlib.sha256(seed_material).hexdigest()[:12]
            aid = f"aug_{digest}"
            rng = random.Random(int(digest, 16))
            augmented = make_variant(frames, kind, rng)
            duration = write_wav(output_audio / f"{aid}.wav", augmented)
            rows.append({
                "anon_id": aid,
                "label": source_row["label"],
                "split": "hidden",
                "duration_s": round(duration, 2),
            })

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    output_manifest = out / "manifest.csv"
    with output_manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["anon_id", "label", "split", "duration_s"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Source calls: {len(source_rows)} ({args.split})")
    print(f"Generated calls: {len(rows)}")
    print(f"Output: {out}")
    print(f"Manifest: {output_manifest}")


if __name__ == "__main__":
    main()
