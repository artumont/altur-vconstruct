# TODO — Training Pipeline

## Phase 1: Dataset + Preprocessing

- [ ] **`src/dataset.py`** — AudioWindowDataset
  - [ ] Read `data/manifest.csv` → filter by split
  - [ ] Load WAV with soundfile → take channel 0 (caller)
  - [ ] Resample 8kHz → 16kHz with `torchaudio.transforms.Resample`
  - [ ] Chop into 4s windows (64000 samples @ 16kHz) with 50% overlap
  - [ ] Each window = 1 sample: `(tensor[64000], label_int)`
  - [ ] Verify: print total window count per split

## Phase 2: Conversation Features (optional but valuable)

- [ ] **`src/dataset.py`** — ConversationFeatures
  - [ ] Parse `data/turns/<call_id>.json` for each call
  - [ ] Extract: num_turns, avg_turn_duration, max_silence_gap, caller_speaking_ratio, overlap_count
  - [ ] Return `np.array` of shape `(5,)` per call

## Phase 3: SSL Embedding Extraction

- [ ] **`src/model.py`** — SSLEvaluator class
  - [ ] Load `microsoft/wavlm-large` from HuggingFace
  - [ ] Freeze all parameters
  - [ ] `extract(waveform[64000])` → `Tensor[1024]` (mean-pool last hidden state)
  - [ ] `extract_batch(waveforms[B, 64000])` → `Tensor[B, 1024]`
- [ ] **`src/train.py`** — `pre_extract_embeddings()`
  - [ ] Loop over train + val splits
  - [ ] Pass all windows through WavLM → get 1024-dim embeddings
  - [ ] Save to `checkpoints/embeddings_cache/{split}.pt`
  - [ ] ⚠️ First run takes 30-60 min. Subsequent runs load from cache.

## Phase 4: Classifier Head

- [ ] **`src/model.py`** — SpoofClassifier class
  - [ ] Input: 1024-dim (+ 5 conv features if enabled)
  - [ ] `Linear(1029→256) → BatchNorm → ReLU → Dropout(0.3)`
  - [ ] `Linear(256→128) → BatchNorm → ReLU → Dropout(0.3)`
  - [ ] `Linear(128→1) → Sigmoid`
  - [ ] Output: scalar probability `[0, 1]`

## Phase 5: Training Loop

- [ ] **`src/train.py`** — `train()`
  - [ ] Load cached embeddings from `checkpoints/embeddings_cache/`
  - [ ] DataLoader: batch_size=64, shuffle=True
  - [ ] Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
  - [ ] Scheduler: CosineAnnealingLR over 30 epochs
  - [ ] Loss: BCELoss
  - [ ] Early stopping: patience=7 epochs
  - [ ] Save best model to `checkpoints/best_model.pt`

## Phase 6: Evaluation

- [ ] Compute EER (Equal Error Rate) on val set each epoch
- [ ] Track best EER, save model when improved
- [ ] Print per-epoch: loss, accuracy, EER

## Phase 7: Calibration

- [ ] After training, run isotonic regression on val predictions
- [ ] Save calibrator to `checkpoints/calibrator.pt`
- [ ] At inference: raw score → `calibrator.predict(score)` → calibrated confidence

## Phase 8: Config

- [ ] **`configs/baseline.yml`** — all hyperparameters in one place

## Phase 9: Entry Point

- [ ] **`src/__main__.py`** — calls `main()` from train.py
- [ ] **`src/__init__.py`** — exports if needed
- [ ] Run: `python -m src` from `apps/train/`

## Phase 10: Inference Pipeline (for API)

- [ ] Load model + calibrator at startup
- [ ] `POST /detect` endpoint:
  - [ ] Decode base64 WAV
  - [ ] Separate channels → take ch0
  - [ ] Resample 8k→16k
  - [ ] Window into 4s chunks
  - [ ] Extract WavLM embeddings (batch)
  - [ ] Run through classifier
  - [ ] Aggregate window scores (mean)
  - [ ] Apply calibration
  - [ ] Return `{"is_synthetic": bool, "confidence": float}`

## Key Numbers

| Metric | Value |
| -------- | ------- |
| Audio files | 353 (150 human, 203 synthetic) |
| Splits | train: 282, val: 71 |
| Training windows | ~14,000 (after windowing) |
| SSL embedding dim | 1024 |
| Classifier params | ~280k |
| Total input dim | 1029 (with conv features) |
| VRAM needed | ~1.5GB |

## Order of Execution

```
Phase 0  →  Phase 1  →  Phase 2  →  Phase 3  →  Phase 5  →  Phase 7  →  Phase 9
(setup)     (dataset)    (turns)     (extract)    (train)     (calibrate)  (run)
```

Phase 4 (classifier) is written alongside Phase 3.  
Phase 6 (eval) runs inside Phase 5's loop.  
Phase 8 (config) written first or alongside Phase 1.  
Phase 10 (API) is separate — only after training works.
