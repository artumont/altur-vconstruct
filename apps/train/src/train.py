"""Training pipeline — embedding extraction, training loop, evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from src.dataset.extractor import SAMPLE_RATE, WINDOW_SAMPLES, SSLEvaluator
from src.dataset.filter import get_samples
from src.dataset.resampler import Resampler
from src.model import SpoofClassifier

logger = logging.getLogger(__name__)

HOP = WINDOW_SAMPLES // 2  # 50% overlap for windowing


# ── Windowing ──────────────────────────────────────────────────────────────────


def window_waveform(waveform: torch.Tensor) -> list[torch.Tensor]:
    """Chop a resampled waveform into 4s windows with 50%% overlap.

    Args:
        waveform: 1-D tensor at 16 kHz.

    Returns:
        List of 1-D tensors, each of shape (64000,).
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


# ── Embedding Extraction ───────────────────────────────────────────────────────


def pre_extract_embeddings(
    manifest_path: str | Path,
    audio_dir: str | Path,
    turns_dir: str | Path,
    cache_dir: str | Path,
    device: str = "cuda",
    batch_size: int = 32,
) -> None:
    """Extract WavLM embeddings for all splits and cache to disk.

    Args:
        manifest_path: Path to manifest CSV.
        audio_dir: Directory containing WAV files.
        turns_dir: Directory containing per-call JSON turn files.
        cache_dir: Output directory for cached embeddings.
        device: Device for WavLM inference.
        batch_size: Batch size for extraction.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    resampler = Resampler(source_sr=SAMPLE_RATE // 2, target_sr=SAMPLE_RATE)
    ssl = SSLEvaluator(device=device)

    for split in ("train", "val"):
        out_path = cache_dir / f"{split}.pt"
        if out_path.exists():
            logger.info("Skipping %s — already cached at %s", split, out_path)
            continue

        samples = get_samples(manifest_path, split, audio_dir, turns_dir)
        logger.info("Extracting embeddings for %s split (%d calls)...", split, len(samples))

        all_embeddings: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for i, sample in enumerate(samples):
            if (i + 1) % 20 == 0:
                logger.info("  [%s %d/%d]", split, i + 1, len(samples))

            try:
                waveform = resampler.resample_file(sample.wav_path)
            except FileNotFoundError:
                logger.warning("  Skipping %s — WAV not found", sample.anon_id)
                continue

            windows = window_waveform(waveform)
            batch = torch.stack(windows)  # (N, 64000)

            # Extract in chunks
            for j in range(0, batch.shape[0], batch_size):
                chunk = batch[j : j + batch_size]
                embs = ssl.extract_batch(chunk)
                all_embeddings.append(embs.cpu())

            label = 1.0 if sample.label == "synthetic" else 0.0
            all_labels.extend([label] * len(windows))  # type: ignore[arg-type]

        embeddings = torch.cat(all_embeddings, dim=0)  # (N_total, 1024)
        labels = torch.tensor(all_labels, dtype=torch.float32)  # (N_total,)

        torch.save({"embeddings": embeddings, "labels": labels}, out_path)
        logger.info(
            "Saved %s: %d windows, shape=%s", split, embeddings.shape[0], tuple(embeddings.shape)
        )


# ── Evaluation ─────────────────────────────────────────────────────────────────


def compute_eer(
    labels: torch.Tensor,
    scores: torch.Tensor,
) -> float:
    """Compute Equal Error Rate (EER).

    Args:
        labels: Binary ground-truth labels (N,).
        scores: Predicted probabilities (N,).

    Returns:
        EER as a float in [0, 1].
    """
    labels_np = labels.cpu().numpy()
    scores_np = scores.cpu().numpy()

    # Sort by score descending
    sorted_indices = np.argsort(-scores_np)
    sorted_labels = labels_np[sorted_indices]

    n_pos = sorted_labels.sum()
    n_neg = len(sorted_labels) - n_pos

    # Walk thresholds
    tp = 0.0
    fp = 0.0
    best = 1.0
    for label in sorted_labels:
        if label == 1.0:
            tp += 1.0
        else:
            fp += 1.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        fnr = (n_pos - tp) / n_pos if n_pos > 0 else 0.0
        best = min(best, abs(fpr - fnr))

    return best


# ── Training Loop ──────────────────────────────────────────────────────────────


def train(
    cache_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 7,
    device: str = "cuda",
) -> SpoofClassifier:
    """Train the spoof classifier on cached embeddings.

    Args:
        cache_dir: Directory containing {split}.pt embedding caches.
        output_dir: Directory to save best model.
        epochs: Max training epochs.
        batch_size: DataLoader batch size.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        patience: Early stopping patience.
        device: Training device.

    Returns:
        Best trained SpoofClassifier.
    """
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cached data
    train_data = torch.load(cache_dir / "train.pt", weights_only=True)
    val_data = torch.load(cache_dir / "val.pt", weights_only=True)

    train_embs = train_data["embeddings"]
    train_labels = train_data["labels"]
    val_embs = val_data["embeddings"]
    val_labels = val_data["labels"]

    logger.info(
        "Loaded embeddings — train: %s, val: %s",
        tuple(train_embs.shape),
        tuple(val_embs.shape),
    )

    # DataLoaders
    train_ds = TensorDataset(train_embs, train_labels)
    val_ds = TensorDataset(val_embs, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    dev = torch.device(device)
    model = SpoofClassifier(input_dim=train_embs.shape[1]).to(dev)
    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    best_eer = 1.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(dev), y_batch.to(dev)
            optimizer.zero_grad()
            preds = model(x_batch).squeeze(1)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.shape[0]
        train_loss /= len(train_ds)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        all_preds: list[torch.Tensor] = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(dev), y_batch.to(dev)
                preds = model(x_batch).squeeze(1)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * x_batch.shape[0]
                all_preds.append(preds.cpu())
        val_loss /= len(val_ds)

        val_scores = torch.cat(all_preds)
        eer = compute_eer(val_labels, val_scores)
        acc = ((val_scores >= 0.5).float() == val_labels).float().mean().item()

        scheduler.step()

        logger.info(
            "Epoch %2d/%d | loss=%.4f | val_loss=%.4f | acc=%.4f | EER=%.4f | lr=%.2e",
            epoch,
            epochs,
            train_loss,
            val_loss,
            acc,
            eer,
            scheduler.get_last_lr()[0],
        )

        # ── Early stopping ──
        if eer < best_eer:
            best_eer = eer
            best_epoch = epoch
            epochs_no_improve = 0
            save_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), save_path)
            logger.info("  ✓ New best — saved to %s", save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    "Early stopping at epoch %d (no improve for %d epochs)", epoch, patience
                )
                break

    logger.info("Training done. Best EER=%.4f at epoch %d", best_eer, best_epoch)

    # Reload best model
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
    return model
