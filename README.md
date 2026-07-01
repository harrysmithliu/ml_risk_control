# Credit Risk Intelligence Platform

An end-to-end credit risk platform using XGBoost, PyTorch, Snowflake, and Streamlit to predict serious delinquency, address class imbalance, and deliver explainable risk assessments.

> **Status:** Project specification / proof-of-concept implementation. This system is for educational and portfolio demonstration purposes only and must not be used to make real lending decisions.

## Overview

This project builds a reproducible machine learning workflow for estimating the probability that a borrower will experience serious delinquency. It combines local development with a Snowflake-ready data layer and presents model performance, risk drivers, single-applicant scoring, and batch scoring through Streamlit.

The design is intentionally lightweight: it uses a modular monolith and Docker instead of premature microservices or Kubernetes, while keeping clear module boundaries for future API extraction, scheduled retraining, and managed cloud deployment.

## Core Capabilities

- Validated data ingestion with local and Snowflake repository implementations.
- Shared Pandas/Scikit-Learn preprocessing for training and inference.
- Logistic-regression baseline.
- Required XGBoost champion with early stopping and bounded tuning.
- PyTorch MLP challenger.
- Original-distribution, class-weighted, and SMOTE imbalance experiments.
- PR-AUC, ROC-AUC, KS, calibration, lift, and business-cost evaluation.
- Probability calibration and validation-driven threshold selection.
- XGBoost native diagnostics, permutation importance, and SHAP explanations.
- Interactive portfolio, model-performance, single-scoring, and batch-scoring pages.
- Versioned model artifacts and prediction audit metadata.
- Docker packaging and GitHub Actions quality gates.

## Data Source

The primary dataset is Kaggle's [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data). Its target, `SeriousDlqin2yrs`, indicates whether a borrower experienced 90 days past due or worse within two years.

The raw dataset is not stored in Git. Users must obtain it through Kaggle and comply with the competition terms. The dataset is anonymized and does not contain a suitable event timeline, so this project uses a reproducible stratified split rather than claiming true out-of-time validation.

The UCI [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350) dataset may be used as an optional pipeline regression fixture, but it is not the primary project result.

## Technology Stack

| Area | Technology |
|---|---|
| Data processing | Pandas, NumPy |
| Pipelines and evaluation | Scikit-Learn, imbalanced-learn |
| Champion model | XGBoost |
| Challenger model | PyTorch |
| Explainability | SHAP, permutation importance |
| Application and charts | Streamlit, Plotly, Matplotlib/Seaborn |
| Data platform | Snowflake |
| Testing and quality | Pytest, Ruff |
| Packaging and automation | Docker, Docker Compose, GitHub Actions |

## Architecture

```mermaid
flowchart LR
    A["Kaggle CSV or Snowflake RAW"] --> B["Validation and Pandas features"]
    B --> C["Logistic baseline"]
    B --> D["XGBoost champion"]
    B --> E["PyTorch challenger"]
    C --> F["Evaluation and calibration"]
    D --> F
    E --> F
    F --> G["Versioned model bundle"]
    G --> H["Streamlit application"]
    H --> I["Snowflake predictions and monitoring data"]
    F --> J["SHAP and permutation reports"]
```

## Modeling Approach

Accuracy is not used to select the model because serious delinquency is a minority-class event. Average Precision/PR-AUC is the primary model-selection metric, supported by ROC-AUC, KS, recall, precision, F1, Brier score, calibration curves, and an explicit business-cost analysis.

XGBoost is evaluated with the original class distribution, `scale_pos_weight`, and SMOTE. SMOTE is applied only inside training folds; validation and test sets retain the real class distribution. The winning model is calibrated where beneficial, and its operating threshold is selected on validation data rather than defaulting to 0.5.

Native XGBoost gain, weight, and cover are treated as diagnostics only. Final model inspection combines repeated held-out permutation importance with SHAP global and local explanations. Correlated features and explanation stability are analyzed explicitly, and explanations are described as model attribution rather than causality.

## Application Experience

The planned Streamlit application includes:

- **Portfolio Overview:** target balance, data quality, missingness, distributions, and correlations.
- **Model Performance:** model comparison, PR/ROC curves, calibration, confusion matrix, lift, threshold economics, and global explanations.
- **Single Applicant:** validated inputs, calibrated probability, demo risk band, threshold context, and local SHAP explanation.
- **Batch Scoring:** schema-validated CSV upload, downloadable predictions, distribution summary, and optional Snowflake writeback.

The interface does not request names or direct personally identifiable information.

## Snowflake Layout

Snowflake is organized into four logical layers:

| Schema | Purpose |
|---|---|
| `RAW` | Source-aligned records and ingestion metadata |
| `CURATED` | Validated and cleaned cases |
| `FEATURES` | Model-ready feature snapshots |
| `SERVING` | Predictions, model versions, and monitoring inputs |

Local CSV/Parquet and Snowflake access implement the same repository interface so that the application can switch backends through configuration.

## Repository Layout

```text
├── app/                 # Streamlit pages and components
├── configs/             # Versioned model and runtime configuration
├── data/                # Local data; ignored by Git
├── docs/                # Requirements, model card, data dictionary, runbook
├── notebooks/           # Exploration only
├── scripts/             # Download, preparation, training, evaluation, loading
├── sql/                 # Snowflake DDL and monitoring views
├── src/ml_risk_control/ # Production data, feature, model, evaluation, and services
├── tests/               # Unit, integration, fixture, and smoke tests
├── Dockerfile
├── compose.yaml
└── pyproject.toml
```

Production logic belongs in `src/`; notebooks must import shared modules rather than becoming an alternative implementation.

## Planned Local Workflow

Once implementation is complete, the standard workflow will be exposed through Make targets or equivalent commands:

```bash
make setup
make data
make eda
make train
make evaluate
make app
make test
```

Container execution will be available through:

```bash
docker compose up --build
```

Snowflake credentials and other secrets will be supplied externally. `.env.example` will document required variable names without storing credentials.

## Quality and Reproducibility

The final model bundle records the preprocessing pipeline, feature schema, model configuration, calibration state, thresholds, risk bands, metrics, dependency versions, random seeds, training time, and data fingerprint.

CI validates linting, unit and integration tests, a lightweight training smoke test, and the Docker build. Tests do not require live Snowflake credentials by default.

## Exploratory Data Analysis

The current repository includes a reproducible EDA workflow through `make eda`, structured summary output in `artifacts/eda/eda_summary.json`, warning-level data-quality checks in the validation layer, field-level treatment decisions for later feature engineering, and Snowflake DDL coverage for `RAW`, `CURATED`, `FEATURES`, and `SERVING`.

### Current Readout

The first EDA pass confirms three project-shaping facts:

- the target is materially imbalanced at `6.684%`
- missingness is concentrated in `MonthlyIncome` and `NumberOfDependents`
- several numeric fields have heavy tails or suspicious extremes, especially `DebtRatio`, `RevolvingUtilizationOfUnsecuredLines`, and the delinquency count columns

The generated chart outputs are versioned under `reports/figures/eda/`.

Target class balance:

![Target class balance](reports/figures/eda/target_balance.png)

This confirms a strong class imbalance and reinforces the choice of PR-AUC and threshold-aware evaluation over raw accuracy.

Missing value counts:

![Missing value counts](reports/figures/eda/missingness.png)

Missingness is concentrated rather than widespread, which supports explicit imputation plus missing-indicator treatment instead of broad column exclusion.

Selected distributions (clipped at p99 in the rendered charts):

![Selected distributions](reports/figures/eda/selected_distributions.png)

The main readout is that `age` is broadly well-behaved, while `MonthlyIncome`, `DebtRatio`, and `RevolvingUtilizationOfUnsecuredLines` need bounded or transformed treatment before stable modeling.

Delinquency count distributions:

![Delinquency distributions](reports/figures/eda/delinquency_distributions.png)

Most observations are concentrated at zero, but the extreme-value scan also identified a small cluster of implausibly large delinquency counts, which are now surfaced as warning-level data-quality findings.

## Current Model Diagnostics

The current XGBoost workflow now persists evaluation-ready artifacts for validation curves, class-imbalance comparison, native feature importance, and permutation importance. The latest local run kept the `reference` candidate as the selected saved model after comparing it against both a bounded tuning search and an automatic `scale_pos_weight` variant.

The artifact details and interpretation notes are documented in [docs/MODEL_ARTIFACTS.md](docs/MODEL_ARTIFACTS.md). The current versioned model figures are stored under `reports/figures/model/`.

Validation Precision-Recall curve:

![Validation Precision-Recall curve](reports/figures/model/pr_curve_validation.png)

This is the primary model-selection view for the current project because serious delinquency is a minority-class outcome.

Validation ROC curve:

![Validation ROC curve](reports/figures/model/roc_curve_validation.png)

This complements PR-AUC by showing global ranking quality across thresholds, but it is not the primary selection metric.

Native XGBoost feature importance (`gain`):

![Native XGBoost gain importance](reports/figures/model/native_importance_gain.png)

This is retained as a fast diagnostic view, not as the final explanation layer on its own.

Validation permutation importance:

![Validation permutation importance](reports/figures/model/permutation_importance.png)

This provides a more robust held-out importance readout than raw split-count or gain-based views alone.

## Current Training Workflow

The current implementation now covers a full local path from validated raw data to reusable scoring artifacts.

1. **Data validation and repository loading**
   - Raw Kaggle files are validated against explicit file contracts before training.
   - Local repository interfaces provide a clean path toward later Snowflake-backed execution.

2. **Feature preparation**
   - The shared preprocessing pipeline applies numeric coercion, bounded clipping, missing-value handling, and explicit missingness indicators.
   - The same preprocessing logic is reused across training and local inference to reduce train-serving skew.

3. **Model training progression**
   - A logistic-regression baseline was implemented first to establish a simple benchmark and artifact contract.
   - The current champion path uses XGBoost with early stopping, bounded parameter search, and reproducible train/validation/test splits.

4. **Class-imbalance and operating-point analysis**
   - The XGBoost workflow compares the original class distribution, `scale_pos_weight`, and SMOTE variants under a shared validation selection metric.
   - Calibration, threshold selection, and business-cost analysis are persisted as explicit artifacts rather than being left as notebook-only analysis.

5. **Artifact packaging for downstream use**
   - The final local bundle includes the persisted model, feature schema, metric reports, curve data, importance outputs, threshold reports, and calibration metadata.
   - These artifacts are now consumed directly by the local Streamlit application for single-applicant scoring and model diagnostics.

## Current Local Streamlit Demo

The current Stage 6 local app already uses these persisted artifacts to support single-applicant scoring with artifact-backed threshold decisions and lightweight interpretation output.

Low-risk example:

![Low-risk Streamlit example](reports/screenshots/11.png)

This example shows a low predicted delinquency probability, with both the F1-oriented threshold and the cost-oriented threshold remaining in the `Pass` state.

Higher-risk example:

![Higher-risk Streamlit example](reports/screenshots/22.png)

This example shows the same interface responding to a riskier delinquency profile, where the predicted probability moves into the `High` band and both threshold decisions switch to `Flag`.

Batch-scoring example:

![Batch-scoring Streamlit example](reports/screenshots/33.png)

This example shows the Stage 7 batch-scoring flow after a CSV upload, including row counts, probability summary metrics, high-risk share, prediction preview, and downloadable scored output.

## Documentation

Data foundation documents:

- [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md)
- [docs/RUNBOOK.md](docs/RUNBOOK.md)

EDA and feature-design documents:

- [docs/EDA_SUMMARY.md](docs/EDA_SUMMARY.md)
- [docs/FEATURE_TREATMENT_DECISIONS.md](docs/FEATURE_TREATMENT_DECISIONS.md)
- [docs/DATA_QUALITY_RULES.md](docs/DATA_QUALITY_RULES.md)

Model artifact and evaluation documents:

- [docs/MODEL_ARTIFACTS.md](docs/MODEL_ARTIFACTS.md)

Application and local demo documents:

- [docs/STREAMLIT_APP_SPEC.md](docs/STREAMLIT_APP_SPEC.md)
- [docs/LOCAL_INFERENCE_FLOW.md](docs/LOCAL_INFERENCE_FLOW.md)
- [docs/BATCH_SCORING_FLOW.md](docs/BATCH_SCORING_FLOW.md)

## Limitations

- The benchmark dataset is anonymized and lacks a reliable time field.
- Results do not constitute regulatory model validation.
- SHAP and permutation importance explain model behavior, not causal relationships.
- Demo thresholds and risk bands are not lending policy.
- The project must not be used for real customer decisions.
