"""Tests for low-latency inference policy helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_API_SRC = Path(__file__).resolve().parents[2] / "apps" / "api" / "src"
if str(_API_SRC) not in sys.path:
    sys.path.insert(0, str(_API_SRC))

from pipeline.service import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    recenter_probability,
    sample_windows,
)


def test_single_window_selects_highest_energy() -> None:
    """Single-window inference should avoid silent caller windows."""
    windows = [torch.zeros(8), torch.full((8,), 3.0), torch.ones(8)]

    selected = sample_windows(windows, max_windows=1)

    assert selected == [windows[1]]


def test_multi_window_sampling_remains_evenly_spaced() -> None:
    """Multi-window inference should preserve full-call coverage."""
    windows = [torch.full((8,), float(index)) for index in range(5)]

    selected = sample_windows(windows, max_windows=2)

    assert selected == [windows[0], windows[4]]


def test_sample_windows_rejects_zero_limit() -> None:
    """At least one inference window is required."""
    with pytest.raises(ValueError, match="at least 1"):
        sample_windows([torch.ones(8)], max_windows=0)


def test_recenter_probability_maps_threshold_to_half() -> None:
    """Configured threshold should become response decision boundary."""
    assert recenter_probability(0.15, 0.15) == pytest.approx(0.5)
    assert recenter_probability(0.0, 0.15) == 0.0
    assert recenter_probability(1.0, 0.15) == 1.0
    assert recenter_probability(0.40, 0.15) > 0.5


def test_recenter_probability_rejects_invalid_threshold() -> None:
    """Threshold must stay inside open probability interval."""
    with pytest.raises(ValueError, match="decision_threshold"):
        recenter_probability(0.5, 1.0)
