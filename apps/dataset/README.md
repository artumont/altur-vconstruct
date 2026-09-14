# Dataset tools

Dataset preparation and deterministic telephony-style audio augmentation.

Augmentation exists so the classifier keeps its accuracy **no matter which
condition the judge's audio carries**. Three variants are generated per training
call, which takes the 282-call train split to 846 variants (~800+ augmented
training calls) for the round-two fine-tune. Validation audio is never augmented.

Each variant applies one family, all deterministic for a fixed seed:

| Family | Perturbation |
| ------ | ------------ |
| Latency shift | Caller channel delayed 100-1500 ms while the agent keeps its timing |
| Noise + gain | Call-level gain -5..+5 dB plus background noise at 18-30 dB SNR |
| Dropout + clock drift | 2-8 packet-loss gaps (50-300 ms) and 0.97-1.03x speed change |

## Augment training data

Original training audio is not copied; the trainer combines the generated
embeddings with the original `train.pt` embeddings (`train_augmented.pt`).

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
  --audio-dir apps/train/data/audio \
  --out apps/train/data/augmented_train \
  --split train --variants 3 --seed 2026
```

Labels are preserved, so the augmented manifest stays compatible with the
training pipeline. See [../train/configs/round2.yml](../train/configs/round2.yml)
and [../../docs/setup.md](../../docs/setup.md) for the fine-tune that consumes it.

## Augment judge data

Generate judge-only perturbations from validation calls without using them for
training:

```bash
make augment-judge
```

Output is written to `tests/judge/augmented_val/` with `split=hidden`. Score them
against a live endpoint the way the judges will:

```bash
python tests/judge/check_endpoint.py \
  --url http://localhost:8000/detect \
  --manifest tests/judge/augmented_val/manifest.csv \
  --audio-dir tests/judge/augmented_val/audio --split hidden
```

`tests/judge/build_test_set.py` (ASVspoof 2019 LA) and
`tests/judge/build_spanish_test_set.py` (HABLA) build additional external
fixtures in the same format.
