# Model Comparison

This document summarizes the current baseline, XGBoost, and PyTorch challenger results from persisted local artifacts and records the current champion rationale.

## Current Ranking

| Rank | Model | Validation PR-AUC | Validation ROC-AUC | Test PR-AUC | Test ROC-AUC | Test Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | XGBoost Champion Candidate | 0.4195 | 0.8711 | 0.4122 | 0.8717 | 0.0485 |
| 2 | PyTorch MLP Challenger | 0.3976 | 0.8723 | 0.3959 | 0.8692 | 0.1494 |
| 3 | Logistic Regression Baseline | 0.3224 | 0.8278 | 0.3303 | 0.8311 | 0.0531 |

## Current Champion

The current champion remains **XGBoost Champion Candidate**.

Champion rationale:

- XGBoost Champion Candidate ranked first on validation PR-AUC (0.4195)
- XGBoost Champion Candidate remained strongest on test PR-AUC (0.4122)
- XGBoost Champion Candidate showed the lowest test Brier score (0.0485)
- The selected model already includes persisted threshold-selection and cost-analysis artifacts.
- The selected model is already wired into both interactive and batch inference paths.

## Interpretation Notes

- XGBoost Champion Candidate is the current deployment-ready path because it combines the strongest ranking performance with persisted threshold governance and active support in the local interactive and batch inference workflows.
- The PyTorch challenger is useful as a non-tree benchmark, but its current raw-probability calibration and Brier performance lag the XGBoost path even though recall is higher at the default threshold.
- The logistic-regression baseline remains the simplest benchmark and a useful reference point, but it is not competitive enough to replace the current champion.

## Artifact Sources

- `artifacts/baseline/baseline_metrics.json`
- `artifacts/xgboost/xgboost_metrics.json`
- `artifacts/xgboost/threshold_selection_report.json`
- `artifacts/xgboost/cost_analysis_report.json`
- `artifacts/xgboost/calibration_report.json`
- `artifacts/torch/torch_metrics.json`
