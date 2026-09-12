# AGENTS.md — HackMTY26

## Repository purpose

Hackathon project for **Altur Challenge: Detect real vs synthetic voice in phone calls**.

- Binary classification: human vs synthetic caller
- Endpoint: `POST /detect` with stereo WAV (8 kHz, ch0=caller, ch1=agent)
- Returns `{ "is_synthetic": bool, "confidence": float }`

## Repository structure

```
hackmty26/
├── api/                     # FastAPI endpoint (POST /detect)
├── audio/                   # ⚠️ Gitignored — unzip altur-challenge-audio.zip here
├── checkpoints/             # Saved model weights
├── configs/                 # YAML config files (baseline.yml)
├── data/
│   ├── manifest.csv         # anon_id, label, split, duration_s
│   └── turns/               # Per-call JSON with speech segments
├── docs/                    # Project documentation (scope, entregables, arquitectura, datasets)
├── src/                     # Core Python modules
│   ├── __init__.py
│   └── __main__.py
├── pyproject.toml           # Python >=3.11, no dependencies declared yet
├── .python-version          # 3.11
└── .vimspector.json         # Debug configurations (debugpy)
```

## Protected / generated areas

- **`audio/`** — Gitignored. Large files. Do not commit. Unzip from release zip.
- **`checkpoints/`** — Model weights. Do not commit unless explicitly requested.
- **`.env`** — Secrets. Never read, write, or commit.

## Data handling

- `data/manifest.csv` maps `anon_id` → label (`human`/`synthetic`), split (`train`/`val`), `duration_s`
- `data/turns/<anon_id>.json` contains speech segments: `{"turns": [{"channel": 0|1, "start": float, "end": float}]}`
- Audio files: `audio/<anon_id>.wav` — stereo, 8 kHz, 16-bit PCM
- Splits are **speaker-disjoint**: no caller appears in both train and val
- **Privacy**: Human callers used invented personal data. Do not attempt identification. Dataset is HackMTY 2026 only, no redistribution.

## Common commands

```bash
# Run the API server
uvicorn api.main:app --reload

# Test the endpoint
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"audio_base64": "<base64_wav>"}'

# Explore dataset
python -c "import pandas as pd; df=pd.read_csv('data/manifest.csv'); print(df['label'].value_counts()); print(df.groupby('label')['duration_s'].describe())"

# Install dependencies (when pyproject.toml is populated)
pip install -e .

# Run tests (when configured)
pytest -x -v --no-header
```

## Key constraints from challenge rules

- Audio is **8 kHz telephony** (not 16 kHz lab)
- Endpoint must be **live during judging** — container + restart policy recommended
- `confidence` breaks ties — calibrate with Platt/isotonic, don't return arbitrary numbers
- Latency is a scoring criterion — avoid overly heavy models
- Judge runs hidden test set against your endpoint during 15-min defense
- Model, language, framework, hosting are all free — only the contract matters

## Docs locations

Refer to `docs/` for detailed project context:

- `docs/scope.md` — Problem, constraints, signals, risks
- `docs/entregables.md` — Deliverables, scoring criteria, checklist
- `docs/datasets.md` — Available datasets and access links
- `docs/arquitectura.md` — Recommended architecture, models, pipeline
