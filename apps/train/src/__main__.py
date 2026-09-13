"""Training entry point for altur-vconstruct voice anti-spoofing classifier."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml


def main() -> None:
    """Main entry point — extract embeddings and/or train classifier.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config file contains invalid YAML.
    """
    parser = argparse.ArgumentParser(description="altur-vconstruct anti-spoofing trainer")
    parser.add_argument(
        "mode",
        choices=["extract", "extract_augmented", "train", "round2", "calibrate", "all"],
        help="extract = cache base embeddings, extract_augmented = cache augmented train embeddings, train = train classifier, round2 = fine-tune from checkpoint, calibrate = fit isotonic regressor, all = extract + train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)  # type: ignore[arg-type]
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    extract_cfg = cfg["extraction"]

    if args.mode in ("extract", "all"):
        from src.train import pre_extract_embeddings

        pre_extract_embeddings(
            manifest_path=data_cfg["manifest_path"],
            audio_dir=data_cfg["audio_dir"],
            turns_dir=data_cfg["turns_dir"],
            cache_dir=data_cfg["cache_dir"],
            device=extract_cfg["device"],
            batch_size=extract_cfg["batch_size"],
        )

    if args.mode in ("extract_augmented", "round2"):
        from src.train import pre_extract_embeddings

        augmented_cfg = data_cfg.get("augmented")
        if augmented_cfg is None:
            raise ValueError("Config requires data.augmented for augmented extraction")
        pre_extract_embeddings(
            manifest_path=augmented_cfg["manifest_path"],
            audio_dir=augmented_cfg["audio_dir"],
            turns_dir=augmented_cfg.get("turns_dir", data_cfg["turns_dir"]),
            cache_dir=augmented_cfg.get("cache_dir", data_cfg["cache_dir"]),
            device=extract_cfg["device"],
            batch_size=extract_cfg["batch_size"],
            splits=("train",),
            cache_suffix=augmented_cfg.get("cache_suffix", "_augmented"),
        )

    if args.mode in ("train", "round2", "all"):
        from src.train import train

        extra_cache_paths = list(train_cfg.get("extra_cache_paths", []))
        init_checkpoint = train_cfg.get("init_checkpoint")
        if args.mode == "round2" and not extra_cache_paths:
            augmented_cfg = data_cfg["augmented"]
            suffix = augmented_cfg.get("cache_suffix", "_augmented")
            augmented_cache_dir = augmented_cfg.get("cache_dir", data_cfg["cache_dir"])
            extra_cache_paths.append(str(Path(augmented_cache_dir) / f"train{suffix}.pt"))

        train(
            cache_dir=data_cfg["cache_dir"],
            output_dir=cfg["output"]["checkpoint_dir"],
            epochs=train_cfg["epochs"],
            batch_size=train_cfg["batch_size"],
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
            patience=train_cfg["patience"],
            device=train_cfg["device"],
            extra_cache_paths=extra_cache_paths,
            init_checkpoint=init_checkpoint,
        )

    if args.mode == "calibrate":
        from src.calibrate import calibrate

        calibrate(
            cache_dir=data_cfg["cache_dir"],
            checkpoint_dir=cfg["output"]["checkpoint_dir"],
            model_path=cfg["output"]["model_path"],
            device=train_cfg["device"],
        )


if __name__ == "__main__":
    main()
