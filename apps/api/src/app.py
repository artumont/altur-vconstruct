"""FastAPI application — POST /detect voice anti-spoofing endpoint.

Contract:
    POST /detect
    body:  {"audio_base64": "<base64 of stereo 8kHz WAV>"}
    resp:  {"is_synthetic": bool, "confidence": float}

channel 0 = caller (classified), channel 1 = agent (context).
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from pipeline.bundle import bundle  # pyright: ignore[reportMissingImports]
from config import get_settings
from schemas import DetectRequest, DetectResponse, ErrorResponse
from pipeline.service import AudioError, classify_wav_bytes, decode_base64_wav  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

app = FastAPI(title="hackmty26 voice anti-spoofing", version="0.1.0")

_settings = get_settings()


@app.on_event("startup")
def _load_models() -> None:
    """Pre-load models so the first request is fast (latency criterion)."""
    try:
        bundle.ensure()
    except Exception as exc:  # pragma: no cover - startup failure
        logger.error("Failed to load models at startup: %s", exc, exc_info=True)


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Liveness probe.

    Returns:
        - dict[str, str | bool]: Health state of the service
    """
    return {
        "status": "ok",
        "device": _settings.device,
        "dtype": _settings.dtype,
        "ready": bundle.ready,
    }


@app.post(
    "/detect",
    response_model=DetectResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def detect(payload: DetectRequest) -> DetectResponse:
    """Classify the caller (channel 0) of a base64-encoded stereo WAV.

    Args:
        payload: Request with base64 WAV.

    Returns:
        Detection result with calibrated confidence.

    Raises:
        HTTPException: On bad payload, unreadable audio, or missing model.
    """
    try:
        bundle.ensure()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        wav_bytes = decode_base64_wav(payload.audio_base64)
        return classify_wav_bytes(wav_bytes)
    except AudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

