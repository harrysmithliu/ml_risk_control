# Model Artifacts and Explainability Outputs

This document describes the current XGBoost training outputs under `artifacts/xgboost/`, how they should be interpreted, and which files are intended for downstream reporting and Streamlit visualization.

## Scope

The current artifact set covers four areas:

- reproducible model persistence
- evaluation metrics and diagnostic curves
- class-imbalance experimentation with `scale_pos_weight`
- model explainability through both native XGBoost importance and permutation importance

These outputs are designed to remain lightweight while preserving a clean path toward richer model serving and UI integration.

## Artifact inventory

The current training run writes the following files to `artifacts/xgboost/`:

- `xgboost_credit_risk.joblib`
  - persisted model artifact including preprocessing pipeline and fitted classifier
- `feature_schema.json`
  - model-facing feature schema and derived input columns
- `split_metadata.json`
  - train, validation, and test partition summaries
- `xgboost_metrics.json`
  - partition-level metric bundle for train, validation, and test
- `curves.json`
  - plot-ready Precision-Recall and ROC curve payloads
- `learning_curve.json`
  - XGBoost evaluation history across boosting rounds
- `native_feature_importance.json`
  - native XGBoost importance exports for gain, weight, and cover
- `permutation_importance.json`
  - validation-based permutation importance report
- `tuning_results.json`
  - reference candidate, bounded search candidates, class-imbalance experiment results, and selected-candidate decisions
- `run_summary.json`
  - final run metadata, selected model source, and artifact reload check
- `xgboost_config_snapshot.json`
  - persisted configuration snapshot used for the run

## Selected candidate logic

The training workflow currently evaluates multiple candidate paths:

- reference candidate
- `scale_pos_weight` variant
- bounded randomized-search tuning candidates

Candidate selection is driven by a common validation metric:

- primary selection metric: `average_precision`
- direction: maximize

This means all candidate paths are compared under the same ranking rule before deciding which model artifact becomes the final saved output.

## Current class-imbalance treatment

The current implementation adds a lightweight first-pass treatment for class imbalance through `scale_pos_weight`.

Configuration lives in `configs/model_xgb.yaml` under:

- `experiments.class_imbalance.run_scale_pos_weight_variant`
- `experiments.class_imbalance.scale_pos_weight_strategy`
- `experiments.class_imbalance.scale_pos_weight_value`

The default strategy is:

- `auto_from_train_ratio`

Under this mode, the script computes:

- `scale_pos_weight = negative_count / positive_count`

For the latest local run, the recorded training distribution was:

- positive count: `7018`
- negative count: `97982`
- computed `scale_pos_weight`: `13.961527500712453`

The result is written into `tuning_results.json` under:

- `class_imbalance_experiments.scale_pos_weight_variant`

This section includes:

- the effective `scale_pos_weight`
- train distribution statistics
- validation metric summary
- whether the imbalance-aware candidate was selected

For the latest local run, the `scale_pos_weight` variant completed successfully but did not outperform the reference candidate on validation `average_precision`, so the final selected model remained the reference candidate.

## Explainability layers

The project now exposes two complementary explainability layers.

### 1. Native XGBoost importance

File:

- `native_feature_importance.json`

This artifact preserves native XGBoost views:

- `gain`
- `weight`
- `cover`

These are fast and useful for quick inspection, but they may be biased toward high-cardinality or frequently split features.

### 2. Permutation importance

File:

- `permutation_importance.json`

This artifact is designed as a more stable companion to native importance.

Current design choices:

- evaluation partition: validation set
- scoring metric: `average_precision`
- repeats: configurable, currently `5`
- feature domain: raw model input features before missing-indicator expansion

Each feature record includes:

- `feature_name`
- `mean_importance`
- `std_importance`
- `mean_permuted_metric`
- `permuted_metric_values`
- `importance_values`

Interpretation:

- a larger positive `mean_importance` indicates a larger performance drop after shuffling that feature
- a near-zero value suggests the feature contributes little incremental signal under the chosen metric
- a high `std_importance` suggests instability across repeats

For the latest local run, the strongest permutation-importance features were:

1. `NumberOfTimes90DaysLate`
2. `RevolvingUtilizationOfUnsecuredLines`
3. `NumberOfTime30-59DaysPastDueNotWorse`
4. `NumberOfTime60-89DaysPastDueNotWorse`
5. `DebtRatio`

## Curves and visualization-ready files

The project already stores plot-ready curve data in:

- `curves.json`

This artifact is the preferred source for Streamlit or static-report plotting because it avoids recomputing curve arrays during presentation.

Current structure:

- `partitions.train.precision_recall_curve`
- `partitions.train.roc_curve`
- `partitions.validation.precision_recall_curve`
- `partitions.validation.roc_curve`
- `partitions.test.precision_recall_curve`
- `partitions.test.roc_curve`

The following four PNGs can already be produced manually from the current artifact set:

1. validation Precision-Recall curve
   - source: `curves.json`
2. validation ROC curve
   - source: `curves.json`
3. native feature-importance bar chart
   - source: `native_feature_importance.json`
4. permutation-importance bar chart
   - source: `permutation_importance.json`

The repository already includes plotting-capable dependencies such as `matplotlib` and `seaborn`, so a manual notebook, ad hoc Python snippet, or dedicated rendering script can generate these figures today.

## Reload and reproducibility guardrails

The current pipeline also includes a practical artifact integrity check.

File:

- `run_summary.json`

Key guardrail:

- the saved model artifact is reloaded immediately after persistence
- the reloaded artifact must return finite probabilities on the validation partition

This reload result is recorded under:

- `run_summary.reload_check`

This helps prevent a successful training job from silently writing an unusable artifact.

## Recommended downstream usage

Use the following files for downstream consumers:

| Consumer | Primary files |
| --- | --- |
| Streamlit charts | `curves.json`, `native_feature_importance.json`, `permutation_importance.json` |
| Model registry summary | `run_summary.json`, `xgboost_config_snapshot.json`, `feature_schema.json` |
| Experiment review | `tuning_results.json`, `xgboost_metrics.json` |
| Split auditing | `split_metadata.json` |
| Training diagnostics | `learning_curve.json` |

## What is not included yet

The current artifact set does not yet include:

- pre-rendered PNG figures committed by the training script
- SHAP-based explainability outputs
- threshold-recommendation artifacts
- SMOTE-based training variants

Those remain possible extensions, but they are not required for the current Stage 4 model-training closure.
