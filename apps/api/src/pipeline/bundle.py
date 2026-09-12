"""Model bundle — lazy-loaded inference stack: WavLM + classifier + calibrator."""

from __future__ import annotations

import logging

import torch

from config import get_settings

logger = logging.getLogger(__name__)


class ModelBundle:
    """Holds the frozen WavLM encoder, spoof classifier, and calibrator.

    Loads lazily (first request) or eagerly via :meth:`ensure`.
    """

    def __init__(self) -> None:
        self.ready = False
        self.resampler = None
        self.ssl = None
        self.classifier = None
        self.calibrator = None

    def ensure(self) -> None:
        """Load the full stack if not already loaded."""
        if self.ready:
            return
        from models.calibrator import load_calibrator  # pyright: ignore[reportMissingImports]
        from models.extractor import ONNXSSLEvaluator  # pyright: ignore[reportMissingImports]
        from models.classifier import SpoofClassifier  # pyright: ignore[reportMissingImports]
        from audio.resampler import Resampler  # pyright: ignore[reportMissingImports]

        settings = get_settings()

        self.resampler = Resampler(source_sr=8000, target_sr=16000)
        self.ssl = ONNXSSLEvaluator(device=settings.device)

        classifier = SpoofClassifier(input_dim=1024)
        model_path = settings.resolved_model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        state = torch.load(
            model_path, weights_only=True, map_location=settings.device
        )
        classifier.load_state_dict(state)
        classifier.to(settings.device).eval()
        self.classifier = classifier

        calibrator_path = settings.resolved_calibrator_path
        if not calibrator_path.exists():
            raise FileNotFoundError(f"Calibrator not found: {calibrator_path}")
        self.calibrator = load_calibrator(calibrator_path)

        self.ready = True
        logger.info(
            "Model stack loaded — device=%s dtype=%s",
            settings.device,
            settings.dtype,
        )


bundle = ModelBundle()

