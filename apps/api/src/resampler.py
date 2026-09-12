"""Audio resampling — convert 8 kHz telephony audio to 16 kHz for WavLM.

Mirror of ``apps/train/src/dataset/resampler.py``, vendored into the API so
the inference image is self-contained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_wav(wav_path: Path | str) -> tuple[torch.Tensor, int]:
    """Load WAV file via soundfile, return (waveform, sample_rate).

    Args:
        wav_path: Path to WAV file.

    Returns:
        Tuple of (1-D waveform tensor, sample rate).
    """
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim == 1:
        return torch.from_numpy(data), sr
    # Stereo → take channel 0 (caller)
    return torch.from_numpy(np.ascontiguousarray(data[:, 0])), sr


class Resampler:
    """Resample stereo WAV from 8 kHz → 16 kHz, return channel 0 (caller)."""

    def __init__(self, source_sr: int = 8000, target_sr: int = 16000) -> None:
        """Initialize resampler.

        Args:
            source_sr: Source sample rate in Hz.
            target_sr: Target sample rate in Hz.
        """
        self.source_sr = source_sr
        self.target_sr = target_sr
        self._transform = torchaudio.transforms.Resample(
            orig_freq=source_sr,
            new_freq=target_sr,
        )

    def resample_file(self, wav_path: Path | str) -> torch.Tensor:
        """Load WAV and resample channel 0 to target sample rate.

        Args:
            wav_path: Path to stereo WAV file.

        Returns:
            1-D tensor of resampled mono audio (caller channel).

        Raises:
            FileNotFoundError: If WAV file doesn't exist.
            ValueError: If audio is empty after loading.
        """
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV not found: {wav_path}")

        waveform, sr = load_wav(wav_path)

        if waveform.shape[0] == 0:
            raise ValueError(f"Empty audio: {wav_path}")

        # Resample if needed
        if sr != self.target_sr:
            waveform = self._transform(waveform)

        return waveform.squeeze(0)  # (1, samples) → (samples,)

    def resample_tensor(
        self,
        waveform: torch.Tensor,
        source_sr: int,
    ) -> torch.Tensor:
        """Resample an already-loaded waveform tensor.

        Args:
            waveform: 1-D or 2-D tensor of audio samples.
            source_sr: Sample rate of the input tensor.

        Returns:
            Resampled 1-D tensor.
        """
        if source_sr == self.target_sr:
            return waveform.squeeze(0) if waveform.dim() > 1 else waveform

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        resampled = self._transform(waveform)
        return resampled.squeeze(0)