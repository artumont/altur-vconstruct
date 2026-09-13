"""Pydantic request/response schemas for the detect endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field  # pyright: ignore[reportMissingImports]


class DetectRequest(BaseModel):
    """Request payload containing a base64-encoded stereo WAV."""

    audio_base64: str = Field(..., description="Base64-encoded stereo 8 kHz WAV")


class DetectResponse(BaseModel):
    """Result of caller classification."""

    is_synthetic: bool
    confidence: float


class ErrorResponse(BaseModel):
    """Error body."""

    detail: str
