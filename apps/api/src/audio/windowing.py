"""Windowing — chop resampled 16 kHz audio into 3s inference windows.

Inference uses shorter windows than training to reduce WavLM latency. Mean
pooling keeps classifier input shape unchanged, so classifier weights remain
compatible.
"""

from __future__ import annotations

import torch

WINDOW_SAMPLES = 48000  # 3s @ 16 kHz
HOP = WINDOW_SAMPLES // 2  # 50% overlap


def window_waveform(waveform: torch.Tensor) -> list[torch.Tensor]:
    """Chop a resampled waveform into 3s windows with 50% overlap.

    Args:
        waveform: 1-D tensor at 16 kHz.

    Returns:
        List of 1-D tensors, each of shape (48000,).
    """
    n = waveform.shape[0]
    if n < WINDOW_SAMPLES:
        # Pad short audio with zeros
        pad = torch.zeros(WINDOW_SAMPLES - n, dtype=waveform.dtype)
        return [torch.cat([waveform, pad])]
    windows: list[torch.Tensor] = []
    start = 0
    while start + WINDOW_SAMPLES <= n:
        windows.append(waveform[start : start + WINDOW_SAMPLES])
        start += HOP
    return windows
