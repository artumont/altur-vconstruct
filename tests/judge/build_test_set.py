"""Build a judge-compatible test set from ASVspoof 2019 LA (HuggingFace).

Downloads ~100 balanced bonafide/spoof samples, resamples to 8 kHz stereo WAV
(ch0=caller, ch1=agent=duplicate), writes manifest.csv + audio/ matching the
judge script format.

Usage:
    python tests/judge/build_test_set.py --out tests/judge/test_set --n 100
    python tests/judge/check_endpoint.py --url http://localhost:8000/detect \
        --manifest tests/judge/test_set/manifest.csv \
        --audio-dir tests/judge/test_set/audio --split all --n 0
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset


def anon_id_from(original_id: str) -> str:
    """Deterministic anonymised ID: call_ + 12 hex chars."""
    h = hashlib.sha256(original_id.encode()).hexdigest()[:12]
    return f"call_{h}"


def decode_flac_bytes_to_8k(audio_bytes: bytes) -> np.ndarray:
    """Decode FLAC bytes -> 8kHz mono int16 via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as src:
        src.write(audio_bytes)
        src_path = src.name
    tmp_path = src_path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-ar", "8000", "-ac", "1",
             "-sample_fmt", "s16", tmp_path],
            capture_output=True, check=True,
        )
        data, _ = sf.read(tmp_path, dtype="int16")
        return data
    finally:
        for p in (src_path, tmp_path):
            if os.path.exists(p):
                os.unlink(p)


def write_stereo_wav(path: Path, mono_int16: np.ndarray, sr: int = 8000) -> float:
    """Write 16-bit stereo WAV (ch0=ch1=mono). Returns duration seconds."""
    stereo = np.column_stack([mono_int16, mono_int16])
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())
    return len(mono_int16) / sr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="tests/judge/test_set", help="output directory")
    ap.add_argument("--n", type=int, default=100, help="total samples (balanced split)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default=None, help="HF cache dir")
    args = ap.parse_args()

    out = Path(args.out)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    half = args.n // 2

    print("Loading ASVspoof 2019 LA from HuggingFace...")
    ds = load_dataset(
        "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA",
        split="test",
        cache_dir=args.cache_dir,
    )

    # Access raw arrow columns (no audio decode needed)
    table = ds.data
    labels = table.column("label").to_pylist()       # 0=bonafide, 1=spoof
    notes_raw = table.column("notes").to_pylist()    # JSON strings
    audio_col = table.column("audio")                # struct<bytes, path>

    bonafide_idx = [i for i, l in enumerate(labels) if l == 0]
    spoof_idx = [i for i, l in enumerate(labels) if l == 1]

    print(f"  bonafide: {len(bonafide_idx)}, spoof: {len(spoof_idx)}")
    print(f"  sampling {half} from each...")

    sampled_bonafide = rng.choice(bonafide_idx, size=min(half, len(bonafide_idx)), replace=False)
    sampled_spoof = rng.choice(spoof_idx, size=min(half, len(spoof_idx)), replace=False)

    rows = []

    for label_val, indices in [("human", sampled_bonafide), ("synthetic", sampled_spoof)]:
        for idx in indices:
            idx = int(idx)
            note = json.loads(notes_raw[idx])
            orig_id = note["utterance_id"]
            aid = anon_id_from(orig_id)

            # Extract raw bytes from arrow
            audio_struct = audio_col[idx].as_py()
            audio_bytes = audio_struct["bytes"]

            audio_8k = decode_flac_bytes_to_8k(audio_bytes)

            wav_path = audio_dir / f"{aid}.wav"
            dur = write_stereo_wav(wav_path, audio_8k, sr=8000)

            rows.append({
                "anon_id": aid,
                "label": label_val,
                "split": "hidden",
                "duration_s": round(dur, 1),
            })
            print(f"  {aid} {label_val:9s} {dur:6.1f}s  ({orig_id})")

    # Shuffle
    rng.shuffle(rows)

    # Write manifest.csv
    manifest_path = out / "manifest.csv"
    with open(manifest_path, "w") as f:
        f.write("anon_id,label,split,duration_s\n")
        for r in rows:
            f.write(f"{r['anon_id']},{r['label']},{r['split']},{r['duration_s']}\n")

    print(f"\nDone: {len(rows)} samples -> {out}")
    print(f"  manifest: {manifest_path}")
    print(f"  audio:    {audio_dir}/")
    print(f"\nTest with:")
    print(f"  python tests/judge/check_endpoint.py \\")
    print(f"    --url http://localhost:8000/detect \\")
    print(f"    --manifest {manifest_path} \\")
    print(f"    --audio-dir {audio_dir} \\")
    print(f"    --split all --n 0")


if __name__ == "__main__":
    main()
