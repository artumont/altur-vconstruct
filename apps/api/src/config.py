"""Runtime configuration — env-driven via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Server configuration, overridable via env vars / .env file."""

    model_config = SettingsConfigDict(
        env_file=(ROOT / ".env") if (ROOT / ".env").exists() else None,
        env_prefix="",
        extra="ignore",
    )

    # Paths (absolute or relative to this file) — env-overridable per deployment
    model_path: Path | None = None
    calibrator_path: Path | None = None

    # Inference knobs
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16" if torch.cuda.is_available() else "float32"
    extract_batch_size: int = 16
    max_windows: int = 2  # cap on windows scored per call (evenly spaced)

    def _resolve(self, name: str, value: Path | None) -> Path:
        """Pick the first existing candidate path, or fall back to the default.

        Docker image puts checkpoints at /app/model/checkpoints; local repo
        keeps them at apps/train/checkpoints.
        """
        if value is not None:
            if value.is_absolute():
                return value
            return ROOT / value
        candidates = [
            ROOT.parent.parent / "train" / "checkpoints",  # repo: apps/train
            ROOT / "model" / "checkpoints",  # docker image
            ROOT / "train" / "checkpoints",  # local api sibling
        ]
        for cand in candidates:
            if (cand / name).exists():
                return cand / name
        return candidates[0] / name

    @property
    def resolved_model_path(self) -> Path:
        return self._resolve("best_model.pt", self.model_path)

    @property
    def resolved_calibrator_path(self) -> Path:
        return self._resolve("calibrator.joblib", self.calibrator_path)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Resolve dtype string to torch.dtype."""
        return torch.float16 if self.dtype == "float16" else torch.float32


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()

