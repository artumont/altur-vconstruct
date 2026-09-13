"""Integration tests — hit a live /detect endpoint over HTTP.

Assumes the API is already running (e.g. ``docker compose up``).
Set ``API_URL`` env var to override the default ``http://localhost:8000``.
Set ``AUDIO_DIR`` env var to override default ``apps/train/audio``.
"""

from __future__ import annotations

import base64
import csv
import io
import os
import struct
import wave
from pathlib import Path

import httpx2 as httpx
import pytest

BASE_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = float(os.getenv("TEST_TIMEOUT", "60"))

# Resolve paths relative to project root (altur-vconstruct/)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", str(_PROJECT_ROOT / "apps" / "train" / "data" / "audio")))
MANIFEST = Path(os.getenv("MANIFEST", str(_PROJECT_ROOT / "apps" / "train" / "data" / "manifest.csv")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_b64(
    duration_s: float = 2.0,
    sample_rate: int = 8000,
    channels: int = 2,
    silent: bool = True,
) -> str:
    """Return a base64-encoded stereo 8 kHz WAV string."""
    n_frames = int(sample_rate * duration_s)
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            if silent:
                wf.writeframes(b"\x00\x00" * (n_frames * channels))
            else:
                samples = [((i % 256) - 128) * 128 for i in range(n_frames * channels)]
                wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        return base64.b64encode(buf.getvalue()).decode()


def _load_val_samples() -> list[dict]:
    """Load val split samples from manifest. Returns list of {anon_id, label, path}."""
    samples = []
    if not MANIFEST.exists():
        return samples
    with open(MANIFEST) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != "val":
                continue
            wav_path = AUDIO_DIR / f"{row['anon_id']}.wav"
            if wav_path.exists():
                samples.append({
                    "anon_id": row["anon_id"],
                    "label": row["label"],
                    "path": wav_path,
                })
    return samples


def _wav_file_to_b64(path: Path) -> str:
    """Read a WAV file and return base64-encoded bytes."""
    return base64.b64encode(path.read_bytes()).decode()


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def val_samples() -> list[dict]:
    return _load_val_samples()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client: httpx.Client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_body(self, client: httpx.Client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "device" in body
        assert "ready" in body


# ---------------------------------------------------------------------------
# POST /detect — happy path (synthetic)
# ---------------------------------------------------------------------------

class TestDetectHappyPath:
    def test_silent_stereo(self, client: httpx.Client):
        r = client.post("/detect", json={"audio_base64": _make_wav_b64()})
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["is_synthetic"], bool)
        assert isinstance(body["confidence"], float)
        assert 0.0 <= body["confidence"] <= 1.0

    def test_non_silent_stereo(self, client: httpx.Client):
        r = client.post("/detect", json={"audio_base64": _make_wav_b64(silent=False)})
        assert r.status_code == 200
        body = r.json()
        assert "is_synthetic" in body
        assert "confidence" in body

    def test_mono_fallback(self, client: httpx.Client):
        """Mono WAV should fall back to the single channel."""
        r = client.post("/detect", json={"audio_base64": _make_wav_b64(channels=1)})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /detect — val dataset integrity
# ---------------------------------------------------------------------------

class TestValDetection:
    """Send real val audio samples and verify detection matches ground truth."""

    def test_human_calls_detected(self, client: httpx.Client, val_samples: list[dict]):
        """Human callers should be classified as NOT synthetic."""
        human_samples = [s for s in val_samples if s["label"] == "human"]
        assert len(human_samples) > 0, "No human val samples found"

        errors = []
        for sample in human_samples[:10]:  # test up to 10
            b64 = _wav_file_to_b64(sample["path"])
            r = client.post("/detect", json={"audio_base64": b64})
            assert r.status_code == 200, f"HTTP {r.status_code} for {sample['anon_id']}"
            body = r.json()
            if body["is_synthetic"]:
                errors.append(
                    f"{sample['anon_id']}: predicted synthetic (conf={body['confidence']:.3f})"
                )

        if errors:
            pytest.fail(f"Human calls misclassified:\n" + "\n".join(errors))

    def test_synthetic_calls_detected(self, client: httpx.Client, val_samples: list[dict]):
        """Synthetic callers should be classified as synthetic."""
        synth_samples = [s for s in val_samples if s["label"] == "synthetic"]
        assert len(synth_samples) > 0, "No synthetic val samples found"

        errors = []
        for sample in synth_samples[:10]:  # test up to 10
            b64 = _wav_file_to_b64(sample["path"])
            r = client.post("/detect", json={"audio_base64": b64})
            assert r.status_code == 200, f"HTTP {r.status_code} for {sample['anon_id']}"
            body = r.json()
            if not body["is_synthetic"]:
                errors.append(
                    f"{sample['anon_id']}: predicted human (conf={body['confidence']:.3f})"
                )

        if errors:
            pytest.fail(f"Synthetic calls misclassified:\n" + "\n".join(errors))

    def test_confidence_decisive(self, client: httpx.Client, val_samples: list[dict]):
        """Correct predictions should be decisive (away from the 0.5 boundary).

        confidence = P(synthetic):
          - human correct → confidence near 0 (low)
          - synthetic correct → confidence near 1 (high)
        Either way, |conf - 0.5| should be large.
        """
        if not val_samples:
            pytest.skip("No val samples available")

        weak = []
        for sample in val_samples[:10]:
            b64 = _wav_file_to_b64(sample["path"])
            r = client.post("/detect", json={"audio_base64": b64})
            assert r.status_code == 200
            body = r.json()
            is_correct = (
                (body["is_synthetic"] and sample["label"] == "synthetic")
                or (not body["is_synthetic"] and sample["label"] == "human")
            )
            # Decisive = confidence far from the 0.5 decision boundary
            margin = abs(body["confidence"] - 0.5)
            if is_correct and margin < 0.05:
                weak.append(
                    f"{sample['anon_id']}: conf={body['confidence']:.3f} (correct but near boundary)"
                )

        if weak:
            pytest.fail(f"Correct predictions too close to boundary:\n" + "\n".join(weak))


# ---------------------------------------------------------------------------
# POST /detect — error cases
# ---------------------------------------------------------------------------

class TestDetectErrors:
    def test_empty_body(self, client: httpx.Client):
        r = client.post("/detect", json={})
        assert r.status_code == 422

    def test_missing_audio_base64(self, client: httpx.Client):
        r = client.post("/detect", json={"wrong_field": "x"})
        assert r.status_code == 422

    def test_invalid_base64(self, client: httpx.Client):
        r = client.post("/detect", json={"audio_base64": "!!!not-base64%%%"})
        assert r.status_code == 400

    def test_garbage_base64(self, client: httpx.Client):
        garbage = base64.b64encode(b"this is not a wav file at all").decode()
        r = client.post("/detect", json={"audio_base64": garbage})
        assert r.status_code == 400

    def test_empty_wav(self, client: httpx.Client):
        with io.BytesIO() as buf:
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(8000)
                wf.writeframes(b"")
            empty_b64 = base64.b64encode(buf.getvalue()).decode()
        r = client.post("/detect", json={"audio_base64": empty_b64})
        assert r.status_code in (400, 200)

    def test_non_json_body(self, client: httpx.Client):
        r = client.post("/detect", content="raw text", headers={"Content-Type": "text/plain"})
        assert r.status_code == 422

    def test_nonexistent_endpoint(self, client: httpx.Client):
        r = client.post("/nope")
        assert r.status_code == 404

    def test_get_on_detect(self, client: httpx.Client):
        r = client.get("/detect")
        assert r.status_code == 405


# ---------------------------------------------------------------------------
# Contract — response shape
# ---------------------------------------------------------------------------

class TestContract:
    """Verify the response matches the challenge spec exactly."""

    def test_response_keys(self, client: httpx.Client):
        body = client.post("/detect", json={"audio_base64": _make_wav_b64()}).json()
        assert set(body.keys()) == {"is_synthetic", "confidence"}

    def test_confidence_is_rounded(self, client: httpx.Client):
        body = client.post("/detect", json={"audio_base64": _make_wav_b64()}).json()
        conf_str = str(body["confidence"])
        if "." in conf_str:
            assert len(conf_str.split(".")[1]) <= 4
