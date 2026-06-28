# Runbook

## Purpose

This runbook describes how to prepare the local environment, acquire the benchmark dataset, validate raw inputs, run the current test suite, and understand the present implementation boundary of the project.

The project is currently in an early proof-of-concept state. The instructions below focus on the Stage 1 foundation that already exists in the repository.

## Prerequisites

- Python `3.11` or `3.12`
- `pip`
- Git
- Kaggle account with access to the *Give Me Some Credit* competition data

Optional:

- Docker and Docker Compose
- Snowflake credentials for later stages

## Repository Setup

From the project root:

```bash
make setup
```

This command upgrades `pip` and installs the project in editable mode with development dependencies.

If `make` is unavailable, use the equivalent Python commands:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
```

## Environment Configuration

Create a local `.env` file from `.env.example` and adjust values only when needed.

Important defaults:

- `DATA_BACKEND=local`
- `RAW_DATA_DIR=data/raw/GiveMeSomeCredit`
- `TRAIN_FILE=cs-training.csv`
- `TEST_FILE=cs-test.csv`

At the current stage, local CSV mode is the primary supported execution path.

## Dataset Acquisition

### Required Source

The primary source is Kaggle's *Give Me Some Credit* dataset:

- Competition page: <https://www.kaggle.com/competitions/GiveMeSomeCredit/data>

### Manual Download Workflow

1. Sign in to Kaggle.
2. Open the competition data page.
3. Download the dataset archive manually.
4. Extract the files into:

```text
data/raw/GiveMeSomeCredit
```

### Expected Files

The local raw directory should contain at least:

- `cs-training.csv`
- `cs-test.csv`

Recommended additional files:

- `sampleEntry.csv`
- `Data Dictionary.xls`

### Git Safety

Raw data must not be committed to Git. The repository `.gitignore` is configured to keep local raw files untracked.

## Raw Data Validation

Run the validation entrypoint:

```bash
make data
```

Equivalent direct command:

```bash
python3 scripts/validate_raw_data.py
```

Successful validation produces:

- console status output
- a JSON report at `artifacts/validation/raw_data_validation_report.json`
- a dataset fingerprint for reproducibility tracking

Current validation checks include:

- required file existence
- expected columns and column order
- optional placeholder-column handling for `cs-test.csv`
- duplicate column detection
- identifier uniqueness
- binary target validation for training data
- missing-value summary

## Tests

Run the current unit test suite:

```bash
pytest tests/unit/test_validation.py tests/unit/test_repositories.py
```

Or use the broader project target:

```bash
make test
```

At the current stage, the most important implemented tests cover:

- raw data contracts
- raw data validation behavior
- local repository loading behavior
- repository factory selection

## Current Implemented Commands

The following commands are currently meaningful and expected to work:

```bash
make setup
make help
make smoke
make data
make test
```

The following targets are already defined but depend on later-stage files that are not yet implemented:

```bash
make train
make evaluate
make app
```

If these commands are executed too early, they fail with explicit entrypoint-missing messages by design.

## Current Stage 1 Artifacts

The following foundation components are already present:

- `pyproject.toml`
- `.env.example`
- `Makefile`
- `src/ml_risk_control/config.py`
- `src/ml_risk_control/data/contracts.py`
- `src/ml_risk_control/data/validation.py`
- `src/ml_risk_control/data/repositories.py`
- `scripts/validate_raw_data.py`
- `docs/DATA_DICTIONARY.md`
- unit tests for validation and repositories

## Data Interpretation Notes

- `Unnamed: 0` is treated as a source identifier, not a model feature.
- `SeriousDlqin2yrs` is the binary target for training workflows only.
- The local `cs-test.csv` file may contain an empty `SeriousDlqin2yrs` placeholder column and must still be treated as unlabeled.
- `MonthlyIncome` and `NumberOfDependents` contain missing values and will require explicit preprocessing later.
- The target is imbalanced, so downstream model selection must not rely on accuracy alone.

## Known Limitations

- The dataset is anonymized and is suitable for benchmarking, not real lending operations.
- No reliable time field is available for true out-of-time validation.
- Snowflake support is scaffolded but not yet operationally validated in this stage.
- Training, evaluation, and Streamlit application entrypoints are not yet implemented.

## Troubleshooting

### `make data` fails

Check the following:

- the raw files exist in `data/raw/GiveMeSomeCredit`
- the file names match the expected defaults
- the CSV columns were not edited locally

Then rerun:

```bash
python3 scripts/validate_raw_data.py
```

### `make test` fails because of missing dependencies

Reinstall the environment:

```bash
make setup
```

### Snowflake repository cannot connect

This is expected unless Snowflake credentials are configured and the required raw tables exist. Local mode should remain the default path during the current stage.
