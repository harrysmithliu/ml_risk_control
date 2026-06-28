PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
STREAMLIT ?= streamlit

APP_ENTRY ?= app/Home.py
DATA_SCRIPT ?= scripts/validate_raw_data.py
TRAIN_SCRIPT ?= scripts/train_xgboost.py
EVALUATE_SCRIPT ?= scripts/evaluate_models.py

.DEFAULT_GOAL := help

.PHONY: help setup install lint format test smoke data train evaluate app clean

help:
	@printf "Available targets:\n"
	@printf "  setup      Install the project with development dependencies\n"
	@printf "  install    Alias for setup\n"
	@printf "  lint       Run Ruff checks\n"
	@printf "  format     Run Ruff formatter\n"
	@printf "  test       Run the test suite\n"
	@printf "  smoke      Run a lightweight import smoke test\n"
	@printf "  data       Run raw data validation\n"
	@printf "  train      Run XGBoost model training\n"
	@printf "  evaluate   Run evaluation and reporting\n"
	@printf "  app        Start the Streamlit application\n"
	@printf "  clean      Remove common local caches\n"

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

install: setup

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest

smoke:
	PYTHONPATH=src $(PYTHON) -c "import ml_risk_control; print('Import smoke test passed.')"

data:
	@if [ ! -f "$(DATA_SCRIPT)" ]; then \
		printf "Missing data entrypoint: %s\n" "$(DATA_SCRIPT)"; \
		exit 1; \
	fi
	$(PYTHON) $(DATA_SCRIPT)

train:
	@if [ ! -f "$(TRAIN_SCRIPT)" ]; then \
		printf "Missing training entrypoint: %s\n" "$(TRAIN_SCRIPT)"; \
		exit 1; \
	fi
	$(PYTHON) $(TRAIN_SCRIPT)

evaluate:
	@if [ ! -f "$(EVALUATE_SCRIPT)" ]; then \
		printf "Missing evaluation entrypoint: %s\n" "$(EVALUATE_SCRIPT)"; \
		exit 1; \
	fi
	$(PYTHON) $(EVALUATE_SCRIPT)

app:
	@if [ ! -f "$(APP_ENTRY)" ]; then \
		printf "Missing Streamlit entrypoint: %s\n" "$(APP_ENTRY)"; \
		exit 1; \
	fi
	$(STREAMLIT) run $(APP_ENTRY)

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
