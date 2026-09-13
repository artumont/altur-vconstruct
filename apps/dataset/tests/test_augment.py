from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np

from dataset.augment import augment_dataset  # pyright: ignore[reportMissingImports]


def _write_source(path: Path) -> None:
    frames = np.zeros((8000, 2), dtype="<i2")
    frames[:, 0] = 1000
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(frames.tobytes())


def test_augment_dataset_is_deterministic_and_preserves_labels(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_source(audio_dir / "call_1.wav")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "anon_id,label,split,duration_s\n"
        "call_1,human,train,1.0\n"
        "call_ignored,synthetic,val,1.0\n",
        encoding="utf-8",
    )

    first = tmp_path / "first"
    second = tmp_path / "second"
    assert augment_dataset(manifest, audio_dir, first, variants=3, seed=7) == (1, 3)
    assert augment_dataset(manifest, audio_dir, second, variants=3, seed=7) == (1, 3)

    first_rows = list(csv.DictReader((first / "manifest.csv").open(encoding="utf-8")))
    second_rows = list(csv.DictReader((second / "manifest.csv").open(encoding="utf-8")))
    assert first_rows == second_rows
    assert {row["label"] for row in first_rows} == {"human"}
    assert {row["split"] for row in first_rows} == {"train"}

    for row in first_rows:
        first_audio = first / "audio" / f"{row['anon_id']}.wav"
        second_audio = second / "audio" / f"{row['anon_id']}.wav"
        assert first_audio.read_bytes() == second_audio.read_bytes()
        with wave.open(str(first_audio), "rb") as source:
            assert source.getnchannels() == 2
            assert source.getframerate() == 8000
            assert source.getsampwidth() == 2
