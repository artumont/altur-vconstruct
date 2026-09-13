"""Inference pipeline — base64 WAV bytes → DetectResponse.

Pure logic, no FastAPI dependencies. Reuses trainer modules from ``src``.
"""

from __future__ import annotations

import base64
import io
import logging
import time

import numpy as np
import soundfile as sf
import torch

from config import get_settings
from pipeline.bundle import bundle  # pyright: ignore[reportMissingImports]
from schemas import DetectResponse

logger = logging.getLogger(__name__)


class AudioError(ValueError):
    """Raised when the WAV payload cannot be decoded."""


def sample_windows(windows: list[torch.Tensor], max_windows: int) -> list[torch.Tensor]:
    """Cap window count while retaining useful caller speech.

    A single-window request selects highest-energy caller window instead of
    first window, which may contain silence. Multi-window requests preserve
    evenly spaced sampling for call coverage.

    Args:
        windows: Full window list from :func:`window_waveform`.
        max_windows: Maximum windows to keep. Short calls keep all.

    Returns:
        At most ``max_windows`` selected windows.

    Raises:
        ValueError: If max_windows is less than one.
    """
    if max_windows < 1:
        raise ValueError("max_windows must be at least 1")

    n = len(windows)
    if n <= max_windows:
        return windows
    if max_windows == 1:
        try:
            energies = torch.stack([window.square().mean() for window in windows])
            selected_index = int(torch.argmax(energies).item())
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("Could not select highest-energy window") from exc
        return [windows[selected_index]]

    idx = np.linspace(0, n - 1, max_windows).round().astype(int)
    return [windows[i] for i in idx]


def recenter_probability(probability: float, threshold: float) -> float:
    """Map selected decision threshold to probability 0.5.

    Monotonic odds recentering preserves ranking while allowing class-confidence
    output to retain existing semantics.

    Args:
        probability: Calibrated synthetic probability.
        threshold: Synthetic decision threshold in original probability space.

    Returns:
        Recentered synthetic probability.

    Raises:
        ValueError: If probability or threshold falls outside valid bounds.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if not 0.0 < threshold < 1.0:
        raise ValueError("decision_threshold must be between 0 and 1")

    numerator = probability * (1.0 - threshold)
    denominator = numerator + (1.0 - probability) * threshold
    return numerator / denominator


def classify_wav_bytes(wav_bytes: bytes) -> DetectResponse:
    """Classify the caller (channel 0) of an encoded stereo WAV.

    Args:
        wav_bytes: Raw WAV file bytes.

    Returns:
        Detection result with calibrated confidence.

    Raises:
        AudioError: If audio is corrupted or empty.
    """
    t0 = time.perf_counter()
    bundle.ensure()
    settings = get_settings()

    # Read WAV
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception as exc:
        raise AudioError(f"Unreadable WAV: {exc}") from exc
    if data.size == 0:
        raise AudioError("Empty audio")
    t_wav = time.perf_counter()

    # Channel 0 = caller; mono falls back to the single channel
    if data.ndim == 1:
        caller = torch.from_numpy(np.ascontiguousarray(data))
    else:
        caller = torch.from_numpy(np.ascontiguousarray(data[:, 0]))

    # Resample 8 kHz → 16 kHz
    assert bundle.resampler is not None
    waveform = bundle.resampler.resample_tensor(caller, source_sr=sr)

    # Window into 3s inference chunks (50% overlap)
    from audio.windowing import window_waveform  # pyright: ignore[reportMissingImports]

    windows = window_waveform(waveform)
    windows = sample_windows(windows, settings.max_windows)
    t_window = time.perf_counter()

    # Batch embeddings through frozen WavLM
    assert bundle.ssl is not None
    batches = []
    dtype = settings.torch_dtype
    for i in range(0, len(windows), settings.extract_batch_size):
        chunk = torch.stack(windows[i : i + settings.extract_batch_size]).to(dtype=dtype)
        batches.append(bundle.ssl.extract_batch(chunk).float().cpu())
    embs = torch.cat(batches, dim=0)  # (N, 1024)
    t_embed = time.perf_counter()

    # Classifier scores per window
    assert bundle.classifier is not None
    with torch.no_grad():
        scores = bundle.classifier(embs.to(settings.device)).squeeze(1).cpu().numpy()
    t_classify = time.perf_counter()

    # Aggregate (mean) + calibrate
    try:
        mean_score = float(scores.mean())
        assert bundle.calibrator is not None
        calibrated_probability = float(bundle.calibrator.predict([mean_score])[0])
        synthetic_probability = recenter_probability(
            calibrated_probability,
            settings.decision_threshold,
        )
    except Exception as exc:
        raise AudioError(f"Scoring failed: {exc}") from exc
    t_end = time.perf_counter()

    duration_s = data.shape[0] / sr
    logger.info(
        "detect latency — wav_decode: %.1fms | resample+window: %.1fms | "
        "onnx_embed: %.1fms (%d windows) | classifier: %.1fms | "
        "calibrate: %.1fms | total: %.1fms | audio_duration: %.1fs",
        (t_wav - t0) * 1000,
        (t_window - t_wav) * 1000,
        (t_embed - t_window) * 1000,
        len(windows),
        (t_classify - t_embed) * 1000,
        (t_end - t_classify) * 1000,
        (t_end - t0) * 1000,
        duration_s,
    )
    return DetectResponse(
        is_synthetic=synthetic_probability >= 0.5,
        confidence=round(max(synthetic_probability, 1 - synthetic_probability), 4),
    )


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
