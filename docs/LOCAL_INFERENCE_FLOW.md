# Local Inference Flow

## Purpose

This document explains how to run the current local Streamlit scoring application, which files it depends on, how to execute a single-applicant demonstration flow, and what to check when local inference fails.

The current application is a local artifact-backed demo layer. It does not call a remote model API, and it does not require a live Snowflake connection.

## Scope

The current local inference flow covers:

- loading the persisted Stage 5 XGBoost artifact bundle
- validating a single applicant input record
- scoring the applicant in-process
- displaying probability, threshold decisions, and diagnostic charts in Streamlit

The current flow does not cover:

- batch scoring
- online feedback capture
- live database reads or writes
- production authentication
- cloud deployment

## Required Environment

- Python `3.11` or `3.12`
- project virtual environment with installed dependencies
- local artifact bundle under `artifacts/xgboost/`

Recommended startup path:

```bash
make setup
```

If the environment already exists, activate or use the local virtual environment directly for all app-related commands.

## Required Artifact Files

The current Streamlit app depends on the following files under `artifacts/xgboost/`:

- `xgboost_credit_risk.joblib`
- `feature_schema.json`
- `run_summary.json`
- `xgboost_metrics.json`
- `curves.json`
- `native_feature_importance.json`
- `permutation_importance.json`
- `threshold_selection_report.json`
- `cost_analysis_report.json`
- `calibration_report.json`

If any of these files are missing, the current local app should be treated as not ready to run.

## Current Code Path

The current local inference path is:

```text
streamlit_app.py
  -> LocalXGBoostInferenceService
  -> XGBoostCreditRiskModel.load(...)
  -> artifact-backed preprocessing and classifier inference
  -> threshold and calibration metadata lookup
  -> Streamlit result rendering
```

Important local files:

- `streamlit_app.py`
- `src/ml_risk_control/inference/service.py`
- `tests/unit/test_inference_service.py`
- `docs/STREAMLIT_APP_SPEC.md`

## Launch Command

From the project root:

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

If the virtual environment is already activated, the equivalent command is:

```bash
python -m streamlit run streamlit_app.py
```

The app should start on the default local Streamlit port unless overridden.

## Pre-Launch Validation

Before launching the UI, the following checks are recommended:

### 1. Verify unit tests for the local inference path

```bash
.venv/bin/python -m pytest tests/unit/test_inference_service.py
```

### 2. Verify the broader model utility surface

```bash
.venv/bin/python -m pytest tests/unit/test_xgboost_model.py tests/unit/test_metrics.py
```

### 3. Verify the app entrypoint syntax

```bash
.venv/bin/python -m py_compile streamlit_app.py
```

These checks do not replace launching the app, but they catch most local integration mistakes before the UI step.

## Expected UI Sections

When the app starts successfully, the current single-page experience should show:

- `Application Status`
- `Applicant Input`
- `Risk Result`
- `Model Diagnostics`

The diagnostics section should expose three tab groups:

- `Curves`
- `Explainability`
- `Thresholding`

## Single-Applicant Demo Flow

### Step 1. Launch the app

Run:

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

### Step 2. Open the local Streamlit URL

Open the local URL printed by Streamlit, typically:

```text
http://localhost:8501
```

### Step 3. Review application status

Confirm that the top status section shows:

- selected candidate source
- F1 threshold
- cost threshold
- calibration method

This confirms that the local artifact bundle was loaded successfully.

### Step 4. Populate the applicant form

Two paths are currently supported:

- click `Use Demo Example`
- manually enter applicant values

The current form includes these raw feature fields:

- `RevolvingUtilizationOfUnsecuredLines`
- `age`
- `NumberOfTime30-59DaysPastDueNotWorse`
- `DebtRatio`
- `MonthlyIncome`
- `NumberOfOpenCreditLinesAndLoans`
- `NumberOfTimes90DaysLate`
- `NumberRealEstateLoansOrLines`
- `NumberOfTime60-89DaysPastDueNotWorse`
- `NumberOfDependents`

`MonthlyIncome` and `NumberOfDependents` may be left blank.

### Step 5. Score the applicant

Click:

- `Score Applicant`

The app should render:

- predicted probability
- risk band
- F1 threshold decision
- cost threshold decision
- interpretation text

### Step 6. Review supporting visuals

Open the diagnostics tabs and confirm that the current rendered PNG assets appear:

- validation PR curve
- validation ROC curve
- native gain importance
- permutation importance
- threshold selection chart
- cost analysis chart

## Input Normalization Behavior

The current inference service applies the saved feature schema contract during local scoring.

Current behaviors include:

- rejection of unexpected fields
- rejection of clearly invalid negative values
- rejection of non-integer-like delinquency count inputs
- support for missing `MonthlyIncome`
- support for missing `NumberOfDependents`
- clipping of selected numeric fields according to the saved feature schema

This means the UI should be treated as an input collection layer, while final normalization rules remain centralized in the inference service.

## Interpretation of Returned Outputs

The current local scoring result is designed to expose multiple views of the same prediction rather than only a binary label.

### Predicted Probability

This is the model-estimated probability of serious delinquency.

### Risk Band

The current app uses a lightweight demo-only risk-banding rule:

- `Low`: probability below `0.10`
- `Medium`: probability between `0.10` and the saved F1 threshold
- `High`: probability at or above the saved F1 threshold

These labels are presentation-only and do not represent a real underwriting policy.

### F1 Threshold Decision

This decision uses the validation-selected threshold stored in:

- `threshold_selection_report.json`

### Cost Threshold Decision

This decision uses the cost-oriented threshold stored in:

- `cost_analysis_report.json`

Because the current cost threshold is more conservative than the F1 threshold, the two decisions may disagree for the same applicant.

## Known Runtime Assumptions

The current app assumes:

- the local XGBoost artifact is readable by `joblib`
- the current Python environment includes `streamlit`, `joblib`, `scikit-learn`, and `xgboost`
- the model figure PNG files already exist under `reports/figures/model/`

If the figures are missing, the page should still start, but the diagnostics section will show warnings instead of images.

## Regenerating Diagnostic Images

If the model PNG files are missing or stale, regenerate them with:

```bash
python3 scripts/render_model_figures.py
```

Expected figure outputs:

- `reports/figures/model/pr_curve_validation.png`
- `reports/figures/model/roc_curve_validation.png`
- `reports/figures/model/native_importance_gain.png`
- `reports/figures/model/permutation_importance.png`
- `reports/figures/model/threshold_selection_validation.png`
- `reports/figures/model/cost_analysis_validation.png`

## Troubleshooting

### The app fails to start because `streamlit` is missing

Use the project environment, then reinstall dependencies if needed:

```bash
make setup
```

Then retry:

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

### The app fails to start because `joblib`, `sklearn`, or `xgboost` is missing

This usually means the system interpreter is being used instead of the project virtual environment.

Use:

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

Do not rely on the default system `python3` for this project.

### The app starts but shows an artifact loading error

Check that all required files exist under:

```text
artifacts/xgboost/
```

If the model bundle or JSON files are missing, rerun the training or restore the expected artifacts.

### The app starts but model charts are missing

Regenerate the local figure files:

```bash
python3 scripts/render_model_figures.py
```

### A submitted applicant fails validation

Review the input values for:

- negative counts
- non-numeric text in numeric fields
- fractional values for count-like fields
- missing required fields

The app should display a user-facing validation message rather than failing silently.

### Streamlit shows local warning messages during development

Mild runtime warnings may still appear during active development. Treat them as a cleanup task only if they affect:

- widget behavior
- scoring correctness
- page rendering
- app startup

## Current Completion Standard

The current local inference flow should be treated as healthy when all of the following are true:

- the app starts in the project virtual environment
- the application status panel loads model metadata
- a single applicant can be scored successfully
- threshold decisions and risk band are shown
- diagnostic figures render
- local inference unit tests pass

That is the current practical closure point for the Stage 6 local demo experience.
