"""WavLM embedding extractor — frozen SSL model for audio feature extraction."""

from __future__ import annotations

import torch
from transformers import WavLMModel

MODEL_NAME = "microsoft/wavlm-large"
EMBEDDING_DIM = 1024
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 64000  # 4s @ 16kHz


class SSLEvaluator:
    """Frozen WavLM-large encoder. Extracts 1024-dim embeddings from audio windows."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        """Load WavLM and freeze all parameters.

        Args:
            device: Target device for inference.
        """
        self.device = torch.device(device)
        self.model = WavLMModel.from_pretrained(MODEL_NAME).to(self.device)  # type: ignore[arg-type]
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract embedding from a single 4s window.

        Args:
            waveform: 1-D tensor of shape (64000,) at 16 kHz.

        Returns:
            1-D tensor of shape (1024,).

        Raises:
            ValueError: If waveform length is not 64000.
        """
        if waveform.shape[0] != WINDOW_SAMPLES:
            raise ValueError(f"Expected {WINDOW_SAMPLES} samples, got {waveform.shape[0]}")
        # (1, samples) → model expects 2-D
        inputs = waveform.unsqueeze(0).to(self.device)
        outputs = self.model(inputs)
        # last_hidden_state: (1, T, 1024) → mean-pool over T → (1024,)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0)

    @torch.no_grad()
    def extract_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from a batch of 4s windows.

        Args:
            waveforms: 2-D tensor of shape (B, 64000) at 16 kHz.

        Returns:
            2-D tensor of shape (B, 1024).

        Raises:
            ValueError: If input is not 2-D.
        """
        if waveforms.dim() != 2:
            raise ValueError(f"Expected 2-D input, got {waveforms.dim()}-D")
        inputs = waveforms.to(self.device)
        outputs = self.model(inputs)
        # (B, T, 1024) → mean-pool → (B, 1024)
        return outputs.last_hidden_state.mean(dim=1)
