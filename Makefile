PYTHON ?= .venv/bin/python
PIP ?= $(PYTHON) -m pip
STREAMLIT ?= $(PYTHON) -m streamlit

APP_ENTRY ?= app/Home.py
DATA_SCRIPT ?= scripts/validate_raw_data.py
EDA_SCRIPT ?= scripts/run_eda.py
TRAIN_SCRIPT ?= scripts/train_baseline.py
EVALUATE_SCRIPT ?= scripts/evaluate_models.py

.DEFAULT_GOAL := help

.PHONY: help ensure-venv setup install lint format test smoke data eda train evaluate app clean

help:
	@printf "Available targets:\n"
	@printf "  setup      Install the project into the local virtual environment\n"
	@printf "  install    Alias for setup\n"
	@printf "  lint       Run Ruff checks\n"
	@printf "  format     Run Ruff formatter\n"
	@printf "  test       Run the test suite\n"
	@printf "  smoke      Run a lightweight import smoke test\n"
	@printf "  data       Run raw data validation\n"
	@printf "  eda        Run the initial exploratory data analysis pass\n"
	@printf "  train      Run XGBoost model training\n"
	@printf "  evaluate   Run evaluation and reporting\n"
	@printf "  app        Start the Streamlit application\n"
	@printf "  clean      Remove common local caches\n"

ensure-venv:
	@if [ ! -x "$(PYTHON)" ]; then \
		printf "Missing project virtualenv interpreter: %s\n" "$(PYTHON)"; \
		printf "Create .venv before running this target.\n"; \
		exit 1; \
	fi

setup: ensure-venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

install: setup

lint: ensure-venv
	$(PYTHON) -m ruff check .

format: ensure-venv
	$(PYTHON) -m ruff format .

test: ensure-venv
	$(PYTHON) -m pytest

smoke: ensure-venv
	PYTHONPATH=src $(PYTHON) -c "import ml_risk_control; print('Import smoke test passed.')"

data: ensure-venv
	@if [ ! -f "$(DATA_SCRIPT)" ]; then \
		printf "Missing data entrypoint: %s\n" "$(DATA_SCRIPT)"; \
		exit 1; \
	fi
	$(PYTHON) $(DATA_SCRIPT)

eda: ensure-venv
	@if [ ! -f "$(EDA_SCRIPT)" ]; then \
		printf "Missing EDA entrypoint: %s\n" "$(EDA_SCRIPT)"; \
		exit 1; \
	fi
	$(PYTHON) $(EDA_SCRIPT)

train: ensure-venv
	@if [ ! -f "$(TRAIN_SCRIPT)" ]; then \
		printf "Missing training entrypoint: %s\n" "$(TRAIN_SCRIPT)"; \
		exit 1; \
	fi
	$(PYTHON) $(TRAIN_SCRIPT)

evaluate: ensure-venv
	@if [ ! -f "$(EVALUATE_SCRIPT)" ]; then \
		printf "Missing evaluation entrypoint: %s\n" "$(EVALUATE_SCRIPT)"; \
		exit 1; \
	fi
	$(PYTHON) $(EVALUATE_SCRIPT)

app: ensure-venv
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
