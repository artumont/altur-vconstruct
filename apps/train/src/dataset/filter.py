"""Manifest filtering — load CSV, filter by split, return sample records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import pandas as pd

Label = Literal["human", "synthetic"]
Split = Literal["train", "val"]


@dataclass(frozen=True, slots=True)
class Sample:
    """One audio file record resolved from the manifest."""

    anon_id: str
    label: Label
    split: Split
    duration_s: float
    wav_path: Path
    turns_path: Path


def load_manifest(manifest_path: Path | str) -> pd.DataFrame:
    """Load manifest.csv and validate required columns exist.

    Args:
        manifest_path: Path to the CSV file.

    Returns:
        DataFrame with manifest rows.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(manifest_path)
    required = {"anon_id", "label", "split", "duration_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    return df


def filter_by_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Filter manifest rows by split name.

    Args:
        df: Manifest DataFrame.
        split: Split name to filter on (e.g. "train", "val").

    Returns:
        Filtered DataFrame.

    Raises:
        ValueError: If split name is not present in the manifest.
    """
    if split not in df["split"].unique():
        available = sorted(df["split"].unique())
        raise ValueError(f"Split '{split}' not in manifest. Available: {available}")
    return df[df["split"] == split].reset_index(drop=True)


def build_samples(
    df: pd.DataFrame,
    audio_dir: Path | str,
    turns_dir: Path | str,
) -> list[Sample]:
    """Convert filtered DataFrame into Sample objects with resolved paths.

    Args:
        df: Filtered manifest DataFrame.
        audio_dir: Directory containing WAV files.
        turns_dir: Directory containing per-call JSON turn files.

    Returns:
        List of Sample records.

    Raises:
        ValueError: If duration_s cannot be cast to float.
    """
    audio_dir = Path(audio_dir)
    turns_dir = Path(turns_dir)
    samples: list[Sample] = []
    for rec in df.to_dict(orient="records"):  # type: ignore[list-item]
        anon_id = str(rec["anon_id"])
        dur_val = rec["duration_s"]
        try:
            dur = float(dur_val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid duration_s for {anon_id}: {dur_val!r}") from exc
        samples.append(
            Sample(
                anon_id=anon_id,
                label=cast(Label, rec["label"]),
                split=cast(Split, rec["split"]),
                duration_s=dur,
                wav_path=audio_dir / f"{anon_id}.wav",
                turns_path=turns_dir / f"{anon_id}.json",
            )
        )
    return samples


def get_samples(
    manifest_path: Path | str,
    split: str,
    audio_dir: Path | str,
    turns_dir: Path | str,
) -> list[Sample]:
    """One-shot: load manifest → filter split → build samples.

    Args:
        manifest_path: Path to manifest CSV.
        split: Split name (e.g. "train", "val").
        audio_dir: Directory containing WAV files.
        turns_dir: Directory containing per-call JSON turn files.

    Returns:
        List of Sample records for the requested split.
    """
    df = load_manifest(manifest_path)
    df = filter_by_split(df, split)
    return build_samples(df, audio_dir, turns_dir)


# ── Convenience Stats ─────────────────────────────────────────────────────────


def print_split_stats(manifest_path: Path | str) -> None:
    """Print window count and label distribution per split."""
    df = load_manifest(manifest_path)
    SAMPLE_RATE = 16000  # after resample
    WINDOW_SAMPLES = 64000  # 4s @ 16kHz
    HOP = WINDOW_SAMPLES // 2  # 50% overlap

    for split_name in sorted(df["split"].unique()):
        sub = df[df["split"] == split_name]
        total_windows = 0
        for rec in sub.to_dict(orient="records"):  # type: ignore[list-item]
            try:
                n_samples = int(float(rec["duration_s"]) * SAMPLE_RATE)
            except (TypeError, ValueError):
                continue
            if n_samples < WINDOW_SAMPLES:
                total_windows += 1
            else:
                total_windows += 1 + (n_samples - WINDOW_SAMPLES) // HOP
        labels: dict[str, int] = sub["label"].value_counts().to_dict()  # type: ignore[assignment]
        n_calls = len(sub.index)
        print(
            f"  {split_name:>5}: {n_calls:>4} calls, ~{total_windows:>6} windows, labels={labels}"
        )
