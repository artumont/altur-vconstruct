"""WavLM embedding extractor — ONNX Runtime backend.

Drop-in replacement for extractor.SSLEvaluator that runs WavLM-large
through ONNX Runtime instead of PyTorch. ~2x faster on CPU.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort  # pyright: ignore[reportMissingImports]
import torch

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1024
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 64000  # 4s @ 16 kHz


def _default_onnx_path() -> Path:
    """Resolve ONNX model path — search common locations."""
    from pathlib import Path as _P
    f = _P(__file__).resolve()  # models/extractor.py
    current = f.parent
    for _ in range(10):
        for sub in ("checkpoints", "model/checkpoints"):
            candidate = current / sub / "wavlm-large.onnx"
            if candidate.exists():
                return candidate
        # Check siblings (e.g. apps/train/checkpoints/)
        for sibling in current.parent.iterdir() if current.parent.exists() else []:
            if sibling.is_dir() and sibling != current:
                for sub in ("checkpoints", "model/checkpoints"):
                    candidate = sibling / sub / "wavlm-large.onnx"
                    if candidate.exists():
                        return candidate
        if current.parent == current:
            break
        current = current.parent
    raise FileNotFoundError(f"wavlm-large.onnx not found from {f.parent}")


class ONNXSSLEvaluator:
    """Frozen WavLM-large via ONNX Runtime. Same interface as SSLEvaluator."""

    def __init__(
        self,
        device: str | torch.device = "cpu",
        model_path: str | Path | None = None,
        intra_threads: int = 6,
    ) -> None:
        """Load WavLM ONNX model.

        Args:
            device: Hint for placement. ONNX provider auto-detected.
            model_path: Path to wavlm-large.onnx. Auto-detected if None.
            intra_threads: ORT intra-op parallelism. 6 is sweet-spot on 12-core.
        """
        self.model_path = Path(model_path) if model_path else _default_onnx_path()

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = intra_threads
        so.inter_op_num_threads = 1  # single batch, no inter-op needed

        # Try GPU first, fall back to CPU
        providers = []
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
            self.device = torch.device("cuda")
            logger.info("ONNX using CUDAExecutionProvider")
        else:
            self.device = torch.device("cpu")
            logger.info("ONNX using CPUExecutionProvider (no GPU)")
        providers.append("CPUExecutionProvider")

        self._session = ort.InferenceSession(str(self.model_path), so, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    @torch.no_grad()
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract embedding from a single 4s window.

        Args:
            waveform: 1-D tensor of shape (64000,) at 16 kHz.

        Returns:
            1-D tensor of shape (1024,).
        """
        if waveform.shape[0] != WINDOW_SAMPLES:
            raise ValueError(f"Expected {WINDOW_SAMPLES} samples, got {waveform.shape[0]}")
        emb = self.extract_batch(waveform.unsqueeze(0))
        return emb.squeeze(0)

    @torch.no_grad()
    def extract_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from a batch of 4s windows.

        Args:
            waveforms: 2-D tensor of shape (B, 64000) at 16 kHz.

        Returns:
            2-D tensor of shape (B, 1024).
        """
        if waveforms.dim() != 2:
            raise ValueError(f"Expected 2-D input, got {waveforms.dim()}-D")

        # ONNX expects numpy float32
        if waveforms.is_cuda:
            waveforms = waveforms.cpu()
        arr = waveforms.numpy().astype(np.float32) if isinstance(waveforms, torch.Tensor) else waveforms.astype(np.float32)

        # Run ONNX inference
        raw = self._session.run(None, {self._input_name: arr})
        last_hidden = np.asarray(raw[0])  # (B, T, 1024)

        # Mean-pool over time axis → (B, 1024)
        embeddings: np.ndarray = last_hidden.mean(axis=1)
        return torch.from_numpy(embeddings)
