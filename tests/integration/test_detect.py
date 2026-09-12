"""Integration tests — hit a live /detect endpoint over HTTP.

Assumes the API is already running (e.g. ``docker compose up``).
Set ``API_URL`` env var to override the default ``http://localhost:8000``.
"""

from __future__ import annotations

import base64
import io
import os
import struct
import wave

import httpx2 as httpx
import pytest

BASE_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = float(os.getenv("TEST_TIMEOUT", "30"))


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
                # deterministic non-silence
                samples = [((i % 256) - 128) * 128 for i in range(n_frames * channels)]
                wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)


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
# POST /detect — happy path
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
# POST /detect — error cases
# ---------------------------------------------------------------------------

class TestDetectErrors:
    def test_empty_body(self, client: httpx.Client):
        r = client.post("/detect", json={})
        assert r.status_code == 422  # Pydantic validation

    def test_missing_audio_base64(self, client: httpx.Client):
        r = client.post("/detect", json={"wrong_field": "x"})
        assert r.status_code == 422

    def test_invalid_base64(self, client: httpx.Client):
        r = client.post("/detect", json={"audio_base64": "!!!not-base64%%%"})
        assert r.status_code == 400

    def test_garbage_base64(self, client: httpx.Client):
        """Valid base64 that decodes to non-audio bytes."""
        garbage = base64.b64encode(b"this is not a wav file at all").decode()
        r = client.post("/detect", json={"audio_base64": garbage})
        assert r.status_code == 400

    def test_empty_wav(self, client: httpx.Client):
        """Zero-length WAV payload."""
        with io.BytesIO() as buf:
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(8000)
                wf.writeframes(b"")
            empty_b64 = base64.b64encode(buf.getvalue()).decode()
        r = client.post("/detect", json={"audio_base64": empty_b64})
        assert r.status_code in (400, 200)  # server decides; just no 500

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
        # confidence should have at most 4 decimal places
        conf_str = str(body["confidence"])
        if "." in conf_str:
            assert len(conf_str.split(".")[1]) <= 4
