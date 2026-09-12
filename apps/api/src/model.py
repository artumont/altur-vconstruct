"""Spoof classifier head — MLP on top of frozen WavLM embeddings.

Mirror of ``apps/train/src/model.py``, vendored into the API so the inference
image is self-contained.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpoofClassifier(nn.Module):
    """Binary classifier on top of WavLM embeddings.

    Architecture:
        Linear(1024 → 256) → BatchNorm → ReLU → Dropout
        Linear(256 → 128)  → BatchNorm → ReLU → Dropout
        Linear(128 → 1)   → Sigmoid
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: tuple[int, int] = (256, 128),
        dropout: float = 0.3,
    ) -> None:
        """Initialize classifier head.

        Args:
            input_dim: Input feature dimension (1024 from WavLM).
            hidden_dims: Sizes of hidden layers.
            dropout: Dropout probability.
        """
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input embeddings of shape (B, input_dim).

        Returns:
            Spoof probabilities of shape (B, 1).
        """
        return self.net(x)