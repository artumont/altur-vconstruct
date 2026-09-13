"""Build long Spanish judge fixtures from HABLA.

HABLA contains Spanish bona-fide and spoof clips, usually 2-13 seconds long.
This script downloads selected clips through the Hugging Face rows API, joins
same-label clips with short silence gaps, resamples to 8 kHz, and writes
judge-compatible stereo WAV calls of at least one minute.

Output:
    <out>/manifest.csv
    <out>/audio/*.wav

Usage:
    uv run --project tests python tests/judge/build_spanish_test_set.py \
        --out tests/judge/spanish_test_set --n 100 --duration 60

    python tests/judge/check_endpoint.py \
        --url http://localhost:8000/detect \
        --manifest tests/judge/spanish_test_set/manifest.csv \
        --audio-dir tests/judge/spanish_test_set/audio \
        --split hidden --n 0

Source:
    https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/HABLA
"""

import argparse
import hashlib
import io
import json
import os
import random
import time
import urllib.parse
import urllib.request
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

ROWS_API = "https://datasets-server.huggingface.co/rows"
DATASET = "SpeechAntiSpoofingBenchmarks/HABLA"
CONFIG = "default"
SPLIT = "test"
# HABLA test rows are ordered spoof first, then real. Confirmed through rows API.
SPOOF_RANGE = (0, 70000)
HUMAN_RANGE = (70000, 80816)


def fetch_json(url: str, attempts: int = 3) -> dict:
    """Fetch JSON with small retry backoff."""
    last_error = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception as exc:  # network errors vary by urllib platform
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def fetch_rows(offset: int, length: int) -> list[dict]:
    """Fetch metadata and signed audio URLs for a row range."""
    query = urllib.parse.urlencode({
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "offset": offset,
        "length": length,
    })
    payload = fetch_json(f"{ROWS_API}?{query}")
    return [item["row"] for item in payload["rows"]]


def download_audio(url: str, attempts: int = 3) -> bytes:
    """Download one signed HABLA audio URL."""
    last_error = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"Could not download audio: {last_error}")


def decode_resample(audio_bytes: bytes) -> np.ndarray:
    """Decode audio bytes and resample mono waveform to 8 kHz int16."""
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = np.asarray(audio, dtype=np.int16)
    if sample_rate == 8000:
        return audio
    target_len = round(len(audio) * 8000 / sample_rate)
    source_x = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    target_x = np.linspace(0.0, 1.0, target_len, endpoint=False)
    resampled = np.interp(target_x, source_x, audio.astype(np.float32))
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def write_stereo_wav(path: Path, caller: np.ndarray, sample_rate: int = 8000) -> float:
    """Write ch0=Spanish caller and ch1=silence as 16-bit stereo WAV."""
    agent = np.zeros_like(caller)
    stereo = np.column_stack((caller, agent))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(stereo.tobytes())
    return len(caller) / sample_rate


def call_id(label: str, index: int, seed: int) -> str:
    """Create deterministic non-source-identifying ID."""
    digest = hashlib.sha256(f"habla:{label}:{index}:{seed}".encode()).hexdigest()[:12]
    return f"es_call_{digest}"


def load_candidates(row_range: tuple[int, int], count: int, rng: random.Random) -> list[dict]:
    """Collect metadata candidates without downloading audio bytes."""
    start, end = row_range
    offsets = list(range(start, end, 100))
    rng.shuffle(offsets)
    candidates = []
    for offset in offsets[: (count + 99) // 100]:
        rows = fetch_rows(offset, min(100, end - offset))
        candidates.extend(rows)
        if len(candidates) >= count:
            break
    rng.shuffle(candidates)
    return candidates[:count]


def build_calls(
    rows: list[dict],
    label: str,
    n_calls: int,
    duration_s: int,
    audio_dir: Path,
    rng: random.Random,
    seed: int,
) -> list[dict]:
    """Build long calls by joining same-label Spanish clips."""
    target_samples = duration_s * 8000
    gap = np.zeros(1600, dtype=np.int16)  # 200 ms conversational gap
    rows = list(rows)
    rng.shuffle(rows)
    cursor = 0
    output_rows = []

    for index in range(n_calls):
        chunks: list[np.ndarray] = []
        total = 0
        while total < target_samples:
            if cursor >= len(rows):
                raise RuntimeError(
                    f"Not enough {label} candidate clips; increase candidate pool"
                )
            row = rows[cursor]
            cursor += 1
            audio_url = row["audio"][0]["src"]
            clip = decode_resample(download_audio(audio_url))
            if not len(clip):
                continue
            chunks.append(clip)
            total += len(clip) + len(gap)
            chunks.append(gap)

        caller = np.concatenate(chunks)[:target_samples]
        aid = call_id(label, index, seed)
        duration = write_stereo_wav(audio_dir / f"{aid}.wav", caller)
        output_rows.append({
            "anon_id": aid,
            "label": label,
            "split": "hidden",
            "duration_s": round(duration, 1),
        })
        print(f"  {aid} {label:9s} {duration:5.1f}s")

    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tests/judge/spanish_test_set")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--duration", type=int, default=60, help="minimum call duration in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates-per-class", type=int, default=900)
    args = parser.parse_args()
    if args.n < 2 or args.duration < 60:
        parser.error("--n must be >= 2 and --duration must be >= 60")

    out = Path(args.out)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    per_class = args.n // 2

    print("Fetching HABLA Spanish metadata...")
    human_rows = load_candidates(HUMAN_RANGE, args.candidates_per_class, rng)
    spoof_rows = load_candidates(SPOOF_RANGE, args.candidates_per_class, rng)
    print(f"  candidate rows: human={len(human_rows)} synthetic={len(spoof_rows)}")

    rows = build_calls(human_rows, "human", per_class, args.duration, audio_dir, rng, args.seed)
    rows += build_calls(spoof_rows, "synthetic", per_class, args.duration, audio_dir, rng, args.seed)
    rng.shuffle(rows)

    manifest = out / "manifest.csv"
    with manifest.open("w", encoding="utf-8") as output:
        output.write("anon_id,label,split,duration_s\n")
        for row in rows:
            output.write(
                f"{row['anon_id']},{row['label']},{row['split']},{row['duration_s']}\n"
            )

    print(f"Done: {len(rows)} calls -> {out}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
