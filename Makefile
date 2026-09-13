.DEFAULT_GOAL := help

UV ?= uv
PYTEST_ARGS ?= -v
TRAIN_CONFIG ?= configs/baseline.yml
TRAIN_MODE ?= all
API_HOST ?= 0.0.0.0
API_PORT ?= 8000
AUGMENT_OUT ?= apps/train/data/augmented_train
JUDGE_AUGMENT_OUT ?= tests/judge/augmented_val
AUGMENT_VARIANTS ?= 3
AUGMENT_SEED ?= 2026

.PHONY: help sync install test test-dataset lint format check api train augment-train augment-judge extract extract-augmented round2 calibrate round2-calibrate clean

help: ## Show available project actions
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "%-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## Install or update dependencies for every project
	$(UV) sync --project apps/api
	$(UV) sync --project apps/dataset
	$(UV) sync --project apps/train
	$(UV) sync --project tests

install: sync ## Alias for sync

test: test-dataset ## Run dataset and integration tests
	$(UV) run --project tests pytest $(PYTEST_ARGS)

test-dataset: ## Run dataset package tests
	$(UV) run --project apps/dataset --extra dev pytest apps/dataset/tests $(PYTEST_ARGS)

lint: ## Run Ruff checks for application code
	cd apps/api && $(UV) run --extra dev ruff check --no-fix src ../dataset
	cd apps/train && $(UV) run --extra dev ruff check --no-fix src

format: ## Format application code with Ruff
	cd apps/api && $(UV) run --extra dev ruff format src ../dataset
	cd apps/train && $(UV) run --extra dev ruff format src

check: lint test ## Run lint and tests

api: ## Start API server
	cd apps/api && PYTHONPATH=src $(UV) run uvicorn app:app --host $(API_HOST) --port $(API_PORT)

train: ## Run training pipeline; override TRAIN_MODE and TRAIN_CONFIG
	cd apps/train && $(UV) run python -m src $(TRAIN_MODE) --config $(TRAIN_CONFIG)

augment-train: ## Generate train-only augmented audio
	$(UV) run --project apps/dataset python -m dataset \
		--manifest apps/train/data/manifest.csv \
		--audio-dir apps/train/audio \
		--out $(AUGMENT_OUT) \
		--split train \
		--variants $(AUGMENT_VARIANTS) \
		--seed $(AUGMENT_SEED)

augment-judge: ## Generate judge perturbations from untouched validation calls
	$(UV) run --project apps/dataset python -m dataset \
		--manifest apps/train/data/manifest.csv \
		--audio-dir apps/train/audio \
		--out $(JUDGE_AUGMENT_OUT) \
		--split val \
		--output-split hidden \
		--variants $(AUGMENT_VARIANTS) \
		--seed $(AUGMENT_SEED)

extract: ## Extract base training and validation embeddings
	$(MAKE) train TRAIN_MODE=extract

extract-augmented: augment-train ## Extract embeddings for augmented train audio
	$(MAKE) train TRAIN_MODE=extract_augmented TRAIN_CONFIG=configs/round2.yml

round2: augment-train ## Fine-tune from baseline best checkpoint with augmented train data
	$(MAKE) train TRAIN_MODE=round2 TRAIN_CONFIG=configs/round2.yml

calibrate: ## Fit baseline validation confidence calibrator
	$(MAKE) train TRAIN_MODE=calibrate

round2-calibrate: ## Calibrate round-two checkpoint on untouched validation data
	$(MAKE) train TRAIN_MODE=calibrate TRAIN_CONFIG=configs/round2.yml

clean: ## Remove generated Python caches and local tool caches
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
