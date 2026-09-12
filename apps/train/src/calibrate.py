"""Calibration — fit isotonic regressor on val predictions, save to disk."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import torch
from sklearn.isotonic import IsotonicRegression

from src.model import SpoofClassifier
from src.train import compute_eer

logger = logging.getLogger(__name__)


def calibrate(
    cache_dir: str | Path,
    checkpoint_dir: str | Path,
    model_path: str | Path,
    device: str = "cuda",
) -> tuple[float, float, float, float]:
    """Fit isotonic calibration on val-set predictions.

    Args:
        cache_dir: Directory containing {split}.pt embedding caches.
        checkpoint_dir: Directory to save the calibrator.
        model_path: Path to the trained best_model.pt state dict.
        device: Device for classifier inference.

    Returns:
        Tuple of (pre-EER, pre-Brier, post-EER, post-Brier).
    """
    cache_dir = Path(cache_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    val_data = torch.load(cache_dir / "val.pt", weights_only=True)
    val_embs = val_data["embeddings"]
    val_labels = val_data["labels"]
    logger.info("Loaded val embeddings: %s", tuple(val_embs.shape))

    model = SpoofClassifier(input_dim=val_embs.shape[1])
    model.load_state_dict(torch.load(model_path, weights_only=True))  # type: ignore[arg-type]
    dev = torch.device(device)
    model.to(dev).eval()

    with torch.no_grad():
        scores = model(val_embs.to(dev)).squeeze(1).cpu()

    # Metrics before calibration
    pre_eer = compute_eer(val_labels, scores)
    pre_brier = ((scores - val_labels) ** 2).mean().item()

    # Fit isotonic regression: score -> P(synthetic)
    reg = IsotonicRegression(out_of_bounds="clip")
    reg.fit(scores.numpy(), val_labels.numpy())

    cal_scores = torch.tensor(reg.predict(scores.numpy()), dtype=torch.float32)
    post_eer = compute_eer(val_labels, cal_scores)
    post_brier = ((cal_scores - val_labels) ** 2).mean().item()

    # Save fitted calibrator
    out_path = checkpoint_dir / "calibrator.joblib"
    joblib.dump(reg, out_path)
    logger.info("Saved calibrator → %s", out_path)
    logger.info(
        "EER   pre=%.4f post=%.4f | Brier pre=%.4f post=%.4f",
        pre_eer,
        post_eer,
        pre_brier,
        post_brier,
    )
    return pre_eer, pre_brier, post_eer, post_brier


def load_calibrator(path: str | Path) -> IsotonicRegression:
    """Load a saved isotonic calibrator.

    Args:
        path: Path to calibrator.joblib.

    Returns:
        Fitted IsotonicRegression.
    """
    return joblib.load(path)