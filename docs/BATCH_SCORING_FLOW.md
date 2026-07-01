# Batch Scoring Flow

## Purpose

This document defines the current local batch-scoring workflow that follows the single-applicant Streamlit demo. It describes the implemented input contract, validation behavior, scoring outputs, UI flow, and current implementation boundary for batch inference.

The goal is to extend the current local artifact-backed scoring path from one applicant at a time to many applicants in a single upload, while preserving the same preprocessing and model-selection assumptions already used in local single-record inference.

## Scope

The current batch-scoring flow covers:

- CSV upload for multiple applicant records
- schema validation before scoring
- artifact-backed local batch inference
- downloadable prediction output
- lightweight summary metrics for the scored batch

The current flow does not cover:

- real-time API serving
- asynchronous job orchestration
- production writeback to Snowflake
- online monitoring dashboards
- external user authentication

## Relationship to the Current Local App

The current Stage 6 local app already supports:

- single-applicant input
- local artifact loading
- probability scoring
- threshold-based decisions
- model diagnostic visuals

The current batch-scoring implementation builds on the same local inference foundation rather than introducing a separate serving stack.

Target direction:

```text
CSV upload
  -> schema validation
  -> shared preprocessing and model inference
  -> probability and threshold decisions
  -> result summary + downloadable output
```

## Current User Experience

The current batch-scoring experience lets a local user:

1. open the Streamlit app
2. navigate to a batch-scoring section
3. upload a CSV file
4. receive clear schema-validation feedback
5. score all valid records in one action
6. inspect a high-level summary of the batch
7. download a prediction-enriched output file

This should feel like a practical internal review tool rather than a notebook-only utility.

## Input File Contract

The current upload contract is aligned with the raw model feature fields used by the single-applicant flow.

Required input columns:

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

Optional passthrough columns:

- `Unnamed: 0`
- other user-owned reference columns, if explicitly allowed by the future implementation

Columns that must not be required for scoring:

- `SeriousDlqin2yrs`
- derived missing-indicator columns such as `MonthlyIncome_missing`

The batch path should use the same raw feature contract as the single-applicant flow and derive all additional preprocessing features internally.

## Validation Rules

Before scoring, the batch path validates:

- file readability
- CSV shape
- required column presence
- duplicate column names
- obviously incompatible types
- impossible negative values for count-like and ratio-like fields

The same semantic rules already used in the local inference service should apply at row level:

- non-negative constraints where appropriate
- integer-like validation for delinquency count fields
- support for blank `MonthlyIncome`
- support for blank `NumberOfDependents`
- clipping based on the saved feature schema

## Validation Outcomes

The current batch flow distinguishes between:

### 1. File-level blocking errors

Examples:

- missing required columns
- duplicate column names
- unreadable file
- unsupported file extension

These should stop the batch job before scoring starts.

### 2. Row-level validation errors

Examples:

- invalid negative delinquency counts
- non-numeric content in numeric columns
- non-integer-like counts such as `1.5`

The current behavior is:

- fail closed on invalid rows
- show a concise validation summary
- ask the user to correct and re-upload the file

This keeps the first version simpler and reduces ambiguity about partial outputs.

## Current Inference Behavior

The batch path loads the same local artifact bundle already used by single-record scoring:

- `artifacts/xgboost/xgboost_credit_risk.joblib`
- `feature_schema.json`
- `run_summary.json`
- `threshold_selection_report.json`
- `cost_analysis_report.json`
- `calibration_report.json`

For every valid input row, the batch-scoring output includes:

- predicted delinquency probability
- risk band
- F1 threshold decision
- cost threshold decision

The implementation avoids any separate preprocessing logic for batch mode. The same feature contract and scoring path is reused to minimize train-serving skew.

## Current Output File

The downloadable output file preserves the uploaded records and appends prediction-oriented fields.

Current output columns append these scoring fields:

- original uploaded columns
- `predicted_probability`
- `risk_band`
- `predicted_label_f1_threshold`
- `predicted_label_cost_threshold`
- `f1_threshold`
- `cost_threshold`
- `selected_candidate_source`

Optional future metadata columns may include:

- `model_version`
- `schema_version`
- `scored_at_utc`

## Current Batch Summary Readout

The Streamlit batch section shows a compact summary after scoring completes.

Current summary fields include:

- uploaded row count
- successfully scored row count
- average predicted probability
- median predicted probability
- high-risk row count
- high-risk share
- count flagged under F1 threshold
- count flagged under cost threshold

Current visuals include:

- predicted-probability bar view across scored rows
- risk-band distribution bar chart

These outputs should keep the batch view lightweight while making the result set easier to inspect than a raw download alone.

## Current Streamlit UX Structure

The batch-scoring section currently lives in the single-page app, though it could later be extracted into a dedicated page.

The current implementation keeps it in the existing app layout below the single-applicant flow. That keeps the demo lightweight and avoids restructuring the whole app before the batch feature is proven useful.

Current UI blocks:

1. Upload panel
2. Validation summary
3. Batch scoring action
4. Batch result summary
5. Download output control

## Suggested Module Layout

The current codebase now includes a small batch-scoring layer.

Implemented additions:

```text
src/ml_risk_control/inference/batch.py
tests/unit/test_batch_inference.py
```

Current responsibilities:

- `batch.py`
  - uploaded dataframe validation
  - batch dataframe normalization
  - vectorized scoring through the current artifact-backed model
  - prediction output assembly
- `test_batch_inference.py`
  - schema validation behavior
  - valid batch scoring path
  - output-column contract

## Relationship to Later Snowflake Work

The batch-scoring flow is intentionally designed so that local CSV scoring can later evolve into:

- Snowflake-backed feature pulls
- prediction writeback into `SERVING`
- audit logging for model version and scoring time

However, the current implementation remains fully local and does not depend on live Snowflake access.

## First Implementation Boundary

The current Stage 7 boundary is:

- CSV upload only
- local scoring only
- fail-closed validation
- downloadable CSV output
- lightweight result summary

The current implementation does not attempt:

- chunked processing for very large files
- background jobs
- concurrent uploads
- automatic writeback
- API extraction

## Testing Expectations

The batch-scoring flow currently includes or targets the following test coverage:

- required-column validation
- invalid-row rejection
- valid CSV-to-prediction path
- output-column contract verification
- threshold-field propagation

UI-level testing can remain lightweight as long as service-level behavior is well covered.

## Recommended Delivery Sequence

The implemented delivery sequence followed this order:

1. add the batch-scoring service layer
2. add unit tests for batch validation and output shape
3. add the Streamlit batch upload section
4. add downloadable export behavior
5. update README and demo documentation

That sequencing keeps the scoring contract stable before UI behavior becomes more complex.

## Completion Standard

The current batch-scoring implementation should be considered complete when:

- a valid CSV upload can be scored locally
- invalid files fail with clear messages
- prediction outputs can be downloaded
- the batch summary view renders correctly
- the logic is covered by direct tests

That is the practical closure point for the batch-scoring extension that follows the single-applicant local demo.
