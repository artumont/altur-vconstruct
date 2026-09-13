"""Inference pipeline — base64 WAV bytes → DetectResponse.

Pure logic, no FastAPI dependencies. Reuses trainer modules from ``src``.
"""

from __future__ import annotations

import base64
import io
import logging

import numpy as np
import soundfile as sf
import torch

from pipeline.bundle import bundle  # pyright: ignore[reportMissingImports]
from config import get_settings
from schemas import DetectResponse

logger = logging.getLogger(__name__)


class AudioError(ValueError):
    """Raised when the WAV payload cannot be decoded."""


def sample_windows(windows: list[torch.Tensor], max_windows: int) -> list[torch.Tensor]:
    """Cap window count to ``max_windows`` evenly spaced windows.

    Long calls yield many 50%-overlap windows; scoring all of them is the
    dominant latency cost. Averaging a small set of evenly spaced windows
    preserves the aggregate signal at a fraction of the compute.

    Args:
        windows: Full window list from :func:`window_waveform`.
        max_windows: Maximum windows to keep. Short calls keep all.

    Returns:
        At most ``max_windows`` windows, evenly spaced across the call.
    """
    n = len(windows)
    if n <= max_windows:
        return windows
    idx = np.linspace(0, n - 1, max_windows).round().astype(int)
    return [windows[i] for i in idx]


def classify_wav_bytes(wav_bytes: bytes) -> DetectResponse:
    """Classify the caller (channel 0) of an encoded stereo WAV.

    Args:
        wav_bytes: Raw WAV file bytes.

    Returns:
        Detection result with calibrated confidence.

    Raises:
        AudioError: If audio is corrupted or empty.
    """
    bundle.ensure()
    settings = get_settings()

    # Read WAV
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception as exc:
        raise AudioError(f"Unreadable WAV: {exc}") from exc
    if data.size == 0:
        raise AudioError("Empty audio")

    # Channel 0 = caller; mono falls back to the single channel
    if data.ndim == 1:
        caller = torch.from_numpy(np.ascontiguousarray(data))
    else:
        caller = torch.from_numpy(np.ascontiguousarray(data[:, 0]))

    # Resample 8 kHz → 16 kHz
    assert bundle.resampler is not None
    waveform = bundle.resampler.resample_tensor(caller, source_sr=sr)

    # Window into 4s chunks (50% overlap)
    from audio.windowing import window_waveform  # pyright: ignore[reportMissingImports]

    windows = window_waveform(waveform)
    windows = sample_windows(windows, settings.max_windows)

    # Batch embeddings through frozen WavLM
    assert bundle.ssl is not None
    batches = []
    dtype = settings.torch_dtype
    for i in range(0, len(windows), settings.extract_batch_size):
        chunk = torch.stack(windows[i : i + settings.extract_batch_size]).to(dtype=dtype)
        batches.append(bundle.ssl.extract_batch(chunk).float().cpu())
    embs = torch.cat(batches, dim=0)  # (N, 1024)

    # Classifier scores per window
    assert bundle.classifier is not None
    with torch.no_grad():
        scores = bundle.classifier(embs.to(settings.device)).squeeze(1).cpu().numpy()

    # Aggregate (mean) + calibrate
    try:
        mean_score = float(scores.mean())
        assert bundle.calibrator is not None
        confidence = float(bundle.calibrator.predict([mean_score])[0])
    except Exception as exc:
        raise AudioError(f"Scoring failed: {exc}") from exc
    return DetectResponse(is_synthetic=confidence >= 0.5, confidence=round(max(confidence, 1 - confidence), 4))


def decode_base64_wav(encoded: str) -> bytes:
    """Decode a base64 string into raw WAV bytes.

    Args:
        encoded: Base64-encoded audio payload.

    Returns:
        Raw WAV bytes.

    Raises:
        AudioError: If the payload is not valid base64.
    """
    try:
        return base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise AudioError("Invalid base64") from exc

