"""Calibrator loading — isotonic regression fitted at training time."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.isotonic import IsotonicRegression


def load_calibrator(path: str | Path) -> IsotonicRegression:
    """Load a saved isotonic calibrator.

    Args:
        path: Path to calibrator.joblib.

    Returns:
        Fitted IsotonicRegression.
    """
    return joblib.load(path)