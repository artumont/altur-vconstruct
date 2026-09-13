"""Create deterministic telephony-style audio augmentations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 8000
FIELDNAMES = ["anon_id", "label", "split", "duration_s"]


def read_wav(path: Path) -> np.ndarray:
    """Read a stereo PCM16 8 kHz WAV as frames x channels."""
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 2 or source.getframerate() != SAMPLE_RATE:
            raise ValueError(f"Expected stereo 8 kHz WAV: {path}")
        if source.getsampwidth() != 2:
            raise ValueError(f"Expected PCM16 WAV: {path}")
        frames = source.readframes(source.getnframes())
    return np.frombuffer(frames, dtype="<i2").reshape(-1, 2).copy()


def write_wav(path: Path, frames: np.ndarray) -> float:
    """Write stereo PCM16 WAV and return duration in seconds."""
    frames = np.clip(frames, -32768, 32767).astype("<i2", copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
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
    caller_rms = np.sqrt(np.mean(output[:, 0] ** 2))
    snr_db = rng.uniform(18.0, 30.0)
    noise_rms = max(8.0, caller_rms / (10 ** (snr_db / 20.0)))
    noise = np.random.default_rng(rng.randrange(2**32)).normal(
        0.0,
        noise_rms,
        size=output.shape,
    )
    return output + noise


def variant_dropout_speed(frames: np.ndarray, rng: random.Random) -> np.ndarray:
    """Add short packet-loss gaps, then apply small clock-rate variation."""
    output = frames.astype(np.float32).copy()
    for _ in range(rng.randint(2, 8)):
        start = rng.randrange(len(output))
        length = min(rng.randint(50, 300) * SAMPLE_RATE // 1000, len(output) - start)
        output[start : start + length, :] = 0.0
    return respeed(output, rng.uniform(0.97, 1.03))


def make_variant(frames: np.ndarray, kind: int, rng: random.Random) -> np.ndarray:
    """Apply one deterministic perturbation family."""
    if kind == 0:
        return variant_latency(frames, rng)
    if kind == 1:
        return variant_noise_gain(frames, rng)
    return variant_dropout_speed(frames, rng)


def augment_dataset(
    manifest_path: str | Path,
    audio_dir: str | Path,
    out_dir: str | Path,
    split: str = "train",
    output_split: str | None = None,
    variants: int = 3,
    seed: int = 2026,
) -> tuple[int, int]:
    """Materialize deterministic variants for one manifest split.

    Original files are not copied. Use original embedding cache together with
    generated embedding cache during training to retain unmodified examples.

    Args:
        manifest_path: Source manifest CSV.
        audio_dir: Directory containing source WAV files.
        out_dir: Output dataset directory containing ``audio/`` and manifest.
        split: Source split to augment.
        output_split: Split value written to generated manifest.
        variants: Number of perturbation families per source call, from 1 to 3.
        seed: Base seed used to derive stable per-call RNGs.

    Returns:
        Tuple of source-call count and generated-call count.

    Raises:
        ValueError: If no source rows exist or variants is outside 1..3.
        FileNotFoundError: If a source WAV is missing.
    """
    if variants not in (1, 2, 3):
        raise ValueError("variants must be 1, 2, or 3")

    manifest_path = Path(manifest_path)
    audio_dir = Path(audio_dir)
    out_dir = Path(out_dir)
    output_audio = out_dir / "audio"
    output_audio.mkdir(parents=True, exist_ok=True)
    output_split = output_split or split

    with manifest_path.open(newline="", encoding="utf-8") as source:
        source_rows = [row for row in csv.DictReader(source) if row["split"] == split]
    if not source_rows:
        raise ValueError(f"No rows for split {split!r}")

    rows: list[dict[str, str | float]] = []
    for source_row in source_rows:
        source_id = source_row["anon_id"]
        source_path = audio_dir / f"{source_id}.wav"
        frames = read_wav(source_path)
        for kind in range(variants):
            seed_material = f"{seed}:{source_id}:{kind}".encode()
            digest = hashlib.sha256(seed_material).hexdigest()[:12]
            anon_id = f"aug_{digest}"
            rng = random.Random(int(digest, 16))
            duration = write_wav(
                output_audio / f"{anon_id}.wav",
                make_variant(frames, kind, rng),
            )
            rows.append(
                {
                    "anon_id": anon_id,
                    "label": source_row["label"],
                    "split": output_split,
                    "duration_s": round(duration, 2),
                }
            )

    random.Random(seed).shuffle(rows)
    output_manifest = out_dir / "manifest.csv"
    with output_manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return len(source_rows), len(rows)


def main() -> None:
    """Run dataset augmentation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-split", default=None)
    parser.add_argument("--variants", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    source_count, generated_count = augment_dataset(
        manifest_path=args.manifest,
        audio_dir=args.audio_dir,
        out_dir=args.out,
        split=args.split,
        output_split=args.output_split,
        variants=args.variants,
        seed=args.seed,
    )
    print(f"Source calls: {source_count} ({args.split})")
    print(f"Generated calls: {generated_count}")
    print(f"Output: {args.out}")
    print(f"Manifest: {Path(args.out) / 'manifest.csv'}")


if __name__ == "__main__":
    main()
