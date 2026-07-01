# Runbook

## Purpose

This runbook describes the current local operating workflow for the project: environment setup, dataset preparation, model training, artifact rendering, Streamlit startup, and test execution.

The repository is now beyond the initial scaffolding stage. The commands below reflect the currently implemented local model workflow, single-applicant scoring flow, and batch-scoring demo flow.

## Prerequisites

- Python `3.11` or `3.12`
- `pip`
- Git
- Kaggle access to the *Give Me Some Credit* dataset

Optional:

- Docker and Docker Compose
- Snowflake credentials for later platform work

## Repository Setup

From the project root:

```bash
python3 -m venv .venv
make setup
```

This installs the project in editable mode with development dependencies into the local virtual environment.

## Environment Configuration

Create a local `.env` file from `.env.example` when needed:

```bash
cp .env.example .env
```

Default local values already assume:

- `DATA_BACKEND=local`
- `RAW_DATA_DIR=data/raw/GiveMeSomeCredit`
- `TARGET_COLUMN=SeriousDlqin2yrs`
- `ID_COLUMN=Unnamed: 0`

For the current local demo workflow, no Snowflake credentials are required.

## Dataset Acquisition

Download Kaggle's *Give Me Some Credit* files and extract them into:

```text
data/raw/GiveMeSomeCredit
```

Expected local files:

- `cs-training.csv`
- `cs-test.csv`

Recommended additional files:

- `sampleEntry.csv`
- `Data Dictionary.xls`

Raw benchmark data must remain outside Git; the repository ignore rules are already set up for that.

## Current Command Surface

The current local workflow uses these primary commands:

```bash
make setup
make data
make eda
make train
make evaluate
make app
make test
```

Command meanings:

- `make setup`
  - install project and dev dependencies into `.venv`
- `make data`
  - validate raw Kaggle files and write a validation artifact
- `make eda`
  - regenerate the current EDA summary and static figures
- `make train`
  - run the current XGBoost training and artifact workflow
- `make evaluate`
  - render model figures from saved artifact JSON outputs
- `make app`
  - launch the current Streamlit application
- `make test`
  - run the full automated test suite

Additional useful command:

```bash
make train-baseline
```

This runs the logistic-regression baseline training script.

## Data Validation Workflow

Run:

```bash
make data
```

Equivalent direct entrypoint:

```bash
.venv/bin/python scripts/validate_raw_data.py
```

Successful validation writes:

- `artifacts/validation/raw_data_validation_report.json`

Validation coverage includes:

- required file existence
- expected columns and ordering
- identifier checks
- binary-target checks
- placeholder-target handling for `cs-test.csv`
- missingness summary

## EDA Workflow

Run:

```bash
make eda
```

Equivalent direct entrypoint:

```bash
.venv/bin/python scripts/run_eda.py
```

Key outputs:

- `artifacts/eda/eda_summary.json`
- figures under `reports/figures/eda/`

## Model Training Workflow

Run the current champion workflow:

```bash
make train
```

Equivalent direct entrypoint:

```bash
.venv/bin/python scripts/train_xgboost.py
```

Optional baseline workflow:

```bash
make train-baseline
```

Key XGBoost artifact outputs are written under:

```text
artifacts/xgboost/
```

Important files include:

- `xgboost_credit_risk.joblib`
- `feature_schema.json`
- `run_summary.json`
- `xgboost_metrics.json`
- `threshold_selection_report.json`
- `cost_analysis_report.json`
- `calibration_report.json`

## Figure Rendering Workflow

Run:

```bash
make evaluate
```

Equivalent direct entrypoint:

```bash
.venv/bin/python scripts/render_model_figures.py
```

This regenerates current model PNG outputs under:

```text
reports/figures/model/
```

## Streamlit Application Workflow

Run:

```bash
make app
```

Equivalent direct entrypoint:

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

The current app supports:

- single-applicant scoring
- batch CSV upload scoring
- threshold-aware result presentation
- static diagnostic charts

## Batch Scoring Demo Workflow

The current batch-scoring demo can be exercised from the Streamlit app with:

- a CSV containing the raw feature columns
- optional passthrough reference columns

For a local demo fixture, use:

- `tmp/batch_ui_demo.csv`

The current batch flow should render:

- uploaded row count
- uploaded column count
- scored-row summary metrics
- scored batch preview
- downloadable scored CSV

## Testing Workflow

Run the full current suite:

```bash
make test
```

Equivalent direct entrypoint:

```bash
.venv/bin/python -m pytest
```

Important focused checks:

```bash
.venv/bin/python -m pytest tests/unit/test_inference_service.py tests/unit/test_batch_inference.py
.venv/bin/python -m py_compile streamlit_app.py src/ml_risk_control/inference/__init__.py src/ml_risk_control/inference/batch.py
```

## Docker Workflow

The repository now includes:

- `Dockerfile`
- `compose.yaml`

Recommended preparation:

```bash
make train
make evaluate
```

Current container startup:

```bash
docker compose up --build
```

The Compose workflow mounts these local directories into the container:

- `artifacts/`
- `reports/`
- `data/raw/`

This keeps the container lightweight while still letting the packaged Streamlit app reuse locally generated model artifacts, rendered figures, and any optional raw-data context.

The container launches the Streamlit app on:

- `http://localhost:8501`

## Troubleshooting

### `make app` fails because `streamlit` is missing

Reinstall the project environment:

```bash
make setup
```

### `make train` fails because XGBoost cannot be imported

Reinstall the local environment and verify:

```bash
.venv/bin/python -c "import xgboost; print(xgboost.__version__)"
```

### Batch scoring UI loads but upload results look wrong

Check:

- required feature columns are present
- numeric columns are actually numeric
- count-like fields are integer-like
- optional fields are blank rather than malformed text

Then retry with:

- `tmp/batch_ui_demo.csv`

### `make evaluate` fails

Check that the XGBoost artifact directory already exists and contains:

- `curves.json`
- `native_feature_importance.json`
- `permutation_importance.json`
- `threshold_selection_report.json`
- `cost_analysis_report.json`

### `make test` fails

Run:

```bash
make setup
```

Then rerun:

```bash
make test
```

### Snowflake integration cannot connect

This is expected unless Snowflake credentials are configured and later serving/data-platform work has been completed. The current default supported workflow remains local.
