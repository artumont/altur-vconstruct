# Dataset tools

Dataset preparation and deterministic telephony-style audio augmentation.

## Augment training data

Generate three variants per training call. Original training audio is not copied;
trainer combines generated embeddings with original `train.pt` embeddings.

```bash
make augment-train
```

Output:

```text
apps/train/data/augmented_train/
├── audio/*.wav
└── manifest.csv
```

Run augmentation directly:

```bash
uv run --project apps/dataset python -m dataset \
  --manifest apps/train/data/manifest.csv \
  --audio-dir apps/train/audio \
  --out apps/train/data/augmented_train \
  --split train --variants 3 --seed 2026
```

Augmentation is deterministic for a fixed seed and keeps labels unchanged.
Validation data is never augmented for training.

## Augment judge data

Generate judge-only perturbations from validation calls without using them for
training:

```bash
make augment-judge
```

Output is written to `tests/judge/augmented_val/` with `split=hidden`.
