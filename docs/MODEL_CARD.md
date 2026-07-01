# Model Card

## Model Summary

- Model name: `xgboost_credit_risk`
- Model version: `0.1.0`
- Schema version: `1.0.0`
- Current status: local champion model for the project demo and batch-scoring workflow
- Backend: local artifact bundle
- Primary task: binary classification for serious delinquency risk (`SeriousDlqin2yrs`)

This model card documents the current champion selected from the project's baseline, XGBoost, and PyTorch challenger experiments. It is intended for technical review, portfolio presentation, and local demonstration use.

## Intended Use

### In-scope use

- local experimentation on a public benchmark dataset
- reproducible model-training and evaluation workflow demonstration
- single-record and batch-scoring demo flows in Streamlit
- portfolio-grade discussion of credit-risk modeling tradeoffs

### Out-of-scope use

- real lending, collections, pricing, or underwriting decisions
- production policy deployment without independent validation
- regulatory, legal, or compliance sign-off
- use on personally identifiable customer records without additional controls

## Training Data

- Primary dataset: Kaggle *Give Me Some Credit*
- Target: `SeriousDlqin2yrs`
- Labeled rows: `150,000`
- Positive class count: `10,026`
- Positive class rate: `6.684%`
- Dataset fingerprint: `61f2f7b456e4481a4e84fba49adf22cd9d6915632df5e64df3374301b961c29d`

The dataset is anonymized and widely used as a public benchmark, but it is not a substitute for institution-owned credit data. It does not include a reliable event-time field, so this project uses a reproducible stratified split rather than claiming true out-of-time validation.

## Train / Validation / Test Protocol

- Train rows: `105,000`
- Validation rows: `22,500`
- Test rows: `22,500`
- Split ratio: `70% / 15% / 15%`
- Random state: `42`
- Stratified split: `True`

The primary model-selection metric is Average Precision (PR-AUC), supported by ROC-AUC, KS, Brier score, precision, recall, F1, threshold analysis, and cost analysis. Accuracy is not used as the main decision metric because the target is materially imbalanced.

## Feature Scope

Raw predictive inputs:

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

Derived model inputs:

- `MonthlyIncome_missing`
- `NumberOfDependents_missing`

Total model input feature count: `12`

The preprocessing path applies numeric coercion, bounded clipping, missing-value handling, and explicit missing-indicator creation. The same feature-preparation logic is reused across training and inference to reduce train-serving skew.

## Candidate Comparison Summary

| Model | Validation PR-AUC | Validation ROC-AUC | Test PR-AUC | Test ROC-AUC | Test Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost Champion Candidate | 0.4195 | 0.8711 | 0.4122 | 0.8717 | 0.0485 |
| PyTorch MLP Challenger | 0.3976 | 0.8723 | 0.3959 | 0.8692 | 0.1494 |
| Logistic Regression Baseline | 0.3224 | 0.8278 | 0.3303 | 0.8311 | 0.0531 |

Why XGBoost remains the champion:

- it ranks first on validation PR-AUC
- it remains strongest on test PR-AUC
- it has the best test Brier score among current candidates
- it already supports threshold-selection and cost-analysis artifacts
- it is already wired into both local interactive and batch inference flows

For the full comparison record, see [MODEL_COMPARISON.md](./MODEL_COMPARISON.md).

## Champion Performance Snapshot

Current XGBoost champion metrics at the default probability threshold (`0.5`):

| Split | PR-AUC | ROC-AUC | KS | Brier | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.4195 | 0.8711 | 0.5930 | 0.0485 | 0.6093 | 0.2094 | 0.3117 |
| Test | 0.4122 | 0.8717 | 0.5899 | 0.0485 | 0.6082 | 0.1961 | 0.2966 |

These default-threshold metrics are useful for apples-to-apples comparison, but they should not be treated as the only operating point.

## Operating Thresholds

The project persists two explicit operating-point views for the current champion:

### F1-oriented threshold

- Validation-selected threshold: `0.19963`
- Validation precision: `0.4025`
- Validation recall: `0.5120`
- Validation F1: `0.4507`

### Cost-oriented threshold

- Validation-selected threshold: `0.13705`
- Validation precision: `0.3378`
- Validation recall: `0.6064`
- Validation F1: `0.4339`
- Test average cost per row under the current simple cost matrix: `0.2096`

The cost-oriented threshold is lower because the current local business assumption penalizes false negatives more heavily than false positives.

## Class-Imbalance Handling

The project explicitly evaluated multiple imbalance-aware paths:

- original class distribution
- `scale_pos_weight`
- SMOTE on the training partition only

The latest local comparison kept the `reference` XGBoost candidate as the selected saved model because the imbalance-aware alternatives did not outperform it on validation PR-AUC. The imbalance experiments are still retained as documented evidence rather than being hidden from downstream review.

## Calibration Status

Calibration was evaluated for the current XGBoost champion with:

- strategy: `train_holdout`
- method: `sigmoid`
- calibration holdout size: `20%` of the training partition

Calibration completed successfully, but it is currently treated as an evaluated option rather than the adopted serving default. The latest local review did not justify replacing the raw-probability path in the active demo workflow.

## Explainability Status

The current explainability stack includes:

- native XGBoost importance (`gain`, `weight`, `cover`)
- held-out permutation importance on the validation partition

These outputs support model review, but they are not equivalent to causal explanation. Native feature importance is retained as a quick diagnostic view and can be biased toward frequent or high-cardinality splits. Permutation importance is the stronger default interpretation layer in the current project.

SHAP-based global and local explanation views are planned as a later enhancement and are not yet part of the active champion artifact bundle.

## Deployment and Serving Context

The current champion model is already integrated with:

- local artifact-backed single-record inference
- local batch-scoring inference
- Streamlit-based demo UI
- Docker packaging
- GitHub Actions CI
- reproducible configuration snapshots and metric artifacts

This means the model is deployment-ready for the scope of this project demo, but not production-ready for regulated credit decisioning.

## Limitations and Risks

- The dataset is anonymized and narrower than a real lending data environment.
- No reliable event-time field is available for out-of-time validation.
- Metrics come from a public benchmark and do not prove real portfolio performance.
- The current cost matrix is illustrative, not business-approved policy logic.
- Calibration has been evaluated, but production-grade probability governance is not complete.
- Explanations describe model behavior, not causal relationships.
- The project does not yet include institution-level monitoring, challenger routing, or approval workflows.

## Recommended Usage Notes

- Use PR-AUC and threshold-context views before relying on any single default-threshold metric.
- Treat the cost-oriented threshold as scenario analysis, not as policy.
- Keep dataset and model limitations visible in demos and documentation.
- Do not use this model for real customer decisions.

## Related Documents

- [MODEL_COMPARISON.md](./MODEL_COMPARISON.md)
- [MODEL_ARTIFACTS.md](./MODEL_ARTIFACTS.md)
- [DATA_DICTIONARY.md](./DATA_DICTIONARY.md)
- [RUNBOOK.md](./RUNBOOK.md)
