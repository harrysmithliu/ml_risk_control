# Streamlit Application Specification

## Purpose

This document defines the Stage 6 scope for the local Streamlit application that serves the current credit-risk model artifacts. The application is intended to provide a lightweight but credible demonstration layer on top of the existing offline training workflow.

The app is a local inference and reporting surface, not a production lending system. It must remain consistent with the current artifact contract under `artifacts/xgboost/` and the current feature schema documented in `feature_schema.json`.

## Stage Goal

Stage 6 establishes a locally runnable Streamlit MVP that can:

- accept a single applicant input record
- score that record with the persisted XGBoost artifact
- display probability-driven risk outputs
- expose threshold and cost-analysis context from saved artifacts
- present model diagnostics that were already produced during training

The stage should prioritize a stable demonstration flow over broader platform engineering.

## Non-Goals

The following items are explicitly out of scope for this stage:

- microservice decomposition
- container orchestration changes
- live Snowflake reads or writes
- online feedback capture
- authentication or role-based access control
- batch upload scoring
- SHAP-heavy interactive explanation workflows
- deployment to managed cloud infrastructure

These may be addressed in later stages after the local demo flow is stable.

## User Experience Summary

The Streamlit app should behave like a small internal demo tool for a model reviewer or business stakeholder.

Primary user actions:

- open the app locally
- review a short system status summary
- enter applicant-level numeric inputs
- submit a scoring request
- inspect the returned risk outputs
- review artifact-based charts and model context

Primary user value:

- understand what the current model predicts for a single applicant
- see how the result changes relative to saved decision thresholds
- inspect the major drivers and diagnostic evidence behind the model

## Delivery Model

The Stage 6 application should use a modular-monolith pattern:

- Streamlit as the UI entry point
- local inference service modules under `src/ml_risk_control/`
- direct loading of persisted local artifacts under `artifacts/xgboost/`

This stage should not introduce a separate model-serving API. The app should load artifacts directly in-process.

## Proposed File Layout

The following files should be added during Stage 6:

```text
streamlit_app.py
src/ml_risk_control/inference/__init__.py
src/ml_risk_control/inference/schemas.py
src/ml_risk_control/inference/service.py
src/ml_risk_control/inference/presentation.py
tests/unit/test_inference_service.py
docs/LOCAL_INFERENCE_FLOW.md
```

Suggested responsibilities:

- `streamlit_app.py`
  - application entry point
  - page composition
  - widget wiring
- `schemas.py`
  - typed single-record input contract
  - field defaults and validation constraints
- `service.py`
  - artifact loading
  - single-row dataframe construction
  - probability inference
  - threshold and metadata lookup
- `presentation.py`
  - risk band mapping
  - badge labels
  - text formatting for model outputs
- `test_inference_service.py`
  - artifact loading and single-record scoring tests

## Runtime Inputs

The first Streamlit MVP should expose a single-applicant form using the current raw feature columns:

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

The app should not ask for:

- `Unnamed: 0`
- `SeriousDlqin2yrs`
- missing-indicator fields such as `MonthlyIncome_missing`

Missing-indicator features must be derived automatically by the shared inference logic rather than being exposed to end users.

## Input Validation Rules

The app should validate inputs before inference and keep the rules aligned with the current feature schema and preprocessing assumptions.

Initial validation rules:

- `age`
  - numeric input
  - expected range: `18` to `100`
- `RevolvingUtilizationOfUnsecuredLines`
  - numeric input
  - expected lower bound: `0`
  - values above `1` may be allowed by raw data history but should be clipped or clearly noted
- `DebtRatio`
  - numeric input
  - expected lower bound: `0`
- delinquency count fields
  - integer-like numeric inputs
  - expected lower bound: `0`
- `MonthlyIncome`
  - optional numeric input
  - expected lower bound: `0`
- `NumberOfDependents`
  - optional numeric input
  - expected lower bound: `0`

Validation behavior:

- block clearly invalid values such as negative age or negative delinquency counts
- allow optional blanks for fields that can be missing in the historical dataset
- standardize valid inputs into a one-row pandas dataframe before inference
- keep clipping logic centralized in inference code rather than duplicating it across widgets

## Inference Contract

The application should load the persisted model bundle from:

- `artifacts/xgboost/xgboost_credit_risk.joblib`

It should also consume supporting artifact files:

- `feature_schema.json`
- `run_summary.json`
- `xgboost_metrics.json`
- `curves.json`
- `native_feature_importance.json`
- `permutation_importance.json`
- `threshold_selection_report.json`
- `cost_analysis_report.json`
- `calibration_report.json`

The inference service should expose a compact structured result that includes:

- raw delinquency probability
- F1-oriented threshold
- cost-oriented threshold
- binary decision under F1 threshold
- binary decision under cost threshold
- risk band
- selected candidate source
- calibration status summary

## Output Design

The app should present the scoring result in business-readable terms rather than only raw model output.

Recommended primary outputs:

- predicted probability
- risk band
- decision at validation-selected threshold
- decision at cost-optimized threshold

Recommended secondary outputs:

- selected candidate source from `run_summary.json`
- current primary model-selection metric
- reference note that the saved model uses artifact-based thresholds rather than a default `0.5`

## Risk Banding

The app should define a simple demo-only risk-banding policy in presentation logic.

Recommended first implementation:

- `Low`: probability < `0.10`
- `Medium`: `0.10 <= probability < F1 threshold`
- `High`: probability >= F1 threshold

This approach is preferred for Stage 6 because it is easier to explain in a local demo and avoids mixing two decision frameworks into a single label. The app can still show both threshold decisions separately.

The app must avoid implying that these bands represent a real underwriting policy. They are presentation-only labels for local demonstration.

## Page Layout

The first version should remain a single-page application with four sections.

### 1. Application Status

This section should show:

- model artifact availability
- selected candidate source
- threshold values currently loaded
- calibration status
- short disclaimer that the tool is for demonstration only

### 2. Applicant Input

This section should provide:

- numeric entry widgets for the ten raw feature fields
- optional help text for missing-allowed fields
- a single `Score Applicant` action button
- a `Use Demo Example` helper for fast walkthroughs, if simple to implement

### 3. Risk Result

This section should display:

- delinquency probability
- risk band
- threshold-based decisions
- concise interpretation text

Recommended result cards:

- Probability
- Risk Band
- F1 Threshold Decision
- Cost Threshold Decision

### 4. Model Diagnostics

This section should reuse existing artifacts and show:

- validation PR curve image
- validation ROC curve image
- permutation importance image
- native gain importance image
- threshold selection image
- cost analysis image

The app may start with a subset of these visuals if layout becomes crowded, but the Stage 6 target should support loading all six.

## Caching and Performance

The app should cache artifact loading to avoid repeated disk I/O on every interaction.

Recommended approach:

- cache the loaded model bundle
- cache parsed JSON artifacts
- keep scoring itself uncached for correctness and simplicity

The expected workload is low, so optimization should stay minimal.

## Error Handling

The app should fail clearly and locally when artifacts are missing or incompatible.

Expected error cases:

- model artifact missing
- JSON artifact missing
- malformed user input
- schema mismatch between app input and model feature schema

Expected behavior:

- show a user-visible error message in Streamlit
- avoid silent fallback behavior
- keep technical details available in expandable diagnostics when practical

## Testing Requirements

Stage 6 should add a lightweight test layer focused on the inference service rather than the full UI runtime.

Minimum required coverage:

- artifact loading succeeds with the current local bundle
- a valid single-record input can be transformed into model-ready input
- scoring returns a finite probability
- threshold metadata is attached to the returned result
- invalid inputs are rejected cleanly

UI smoke tests are optional in this stage.

## Documentation Requirements

Stage 6 should conclude with:

- this specification document
- `docs/LOCAL_INFERENCE_FLOW.md`
- README updates describing how to launch the app locally

The README update should remain concise and point readers to the detailed docs rather than duplicating implementation detail.

## Completion Criteria

Stage 6 is considered complete when all of the following are true:

- `streamlit run streamlit_app.py` starts successfully in the project environment
- the app can score a single applicant using the persisted XGBoost artifact
- the app displays probability, risk band, and both saved threshold decisions
- the app loads and displays the existing model diagnostic visuals
- the inference service has direct test coverage
- the run instructions are documented

## Stage 6 Closure Point

Stage 6 should stop once the local single-applicant demonstration flow is stable.

The next stage can then decide whether to extend toward:

- batch scoring
- API extraction
- Snowflake integration
- richer explanation workflows
- Dockerized local serving

That sequencing keeps the current implementation lightweight while preserving a clean future expansion path.
