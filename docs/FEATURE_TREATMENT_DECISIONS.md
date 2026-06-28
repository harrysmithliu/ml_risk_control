# Feature Treatment Decisions

## Purpose

This document records the current field-level treatment decisions for the Stage 2 data foundation. Its purpose is to prevent Stage 3 from reopening basic source-field questions such as whether a column is a feature, how missing values are handled, and which values should be flagged as suspicious.

These decisions are intentionally practical rather than regulatory. They are designed for a reproducible portfolio-grade implementation using Kaggle's *Give Me Some Credit* benchmark.

## Decision Status Legend

- `Locked`: the project should proceed with this treatment unless a clear implementation issue is found
- `Provisional`: the direction is selected, but the exact implementation detail may still change during feature-pipeline coding

## Global Treatment Principles

- `Unnamed: 0` is a source identifier and must never be used as a predictive feature.
- `SeriousDlqin2yrs` is the supervised target and must never appear in scoring-time features.
- Missingness is treated explicitly, not silently ignored.
- Suspicious numeric extremes are first flagged at the validation layer and then handled in the feature pipeline.
- Hard schema failures remain separate from warning-level data-quality issues.
- Stage 3 may implement clipping, indicator columns, and transformations, but it should not revisit whether the source fields belong in scope.

## Field-Level Decisions

| Column | Role | Use In Features | Missingness Treatment | Suspicious / Extreme Value Treatment | Stage 3 Action | Status |
|---|---|---|---|---|---|---|
| `Unnamed: 0` | Source identifier | No | None required | None required | Exclude from modeling dataframes after loading and auditing | Locked |
| `SeriousDlqin2yrs` | Binary target | No | None in training data | Must remain binary `{0, 1}` | Use only for split, training, and evaluation | Locked |
| `RevolvingUtilizationOfUnsecuredLines` | Numeric feature | Yes | No current missingness treatment required | Values above `1.0` remain allowed but must be flagged; long-tail handling is required before model fitting | Keep raw feature, add capped or clipped variant, and consider log-like stabilization only if empirically helpful | Provisional |
| `age` | Numeric feature | Yes | No current missingness treatment required | Values below `18` or above `100` are warning-level anomalies and must be reviewed in feature pipeline | Retain field, add anomaly handling rule, and decide between capping or row exclusion during curated processing | Provisional |
| `NumberOfTime30-59DaysPastDueNotWorse` | Delinquency count feature | Yes | No current missingness treatment required | Values above `20` are suspicious and must be flagged; very large counts should not flow into modeling untreated | Retain field, create capped version, and review whether a binary severe-delinquency indicator is also needed | Provisional |
| `DebtRatio` | Numeric feature | Yes | No current missingness treatment required | Values above `5.0` are warning-level anomalies; the distribution has severe long-tail behavior | Retain field, apply clipping or transformation strategy before model training, and compare raw vs bounded variants | Provisional |
| `MonthlyIncome` | Numeric feature | Yes | Missing values require explicit imputation; non-positive values are warning-level anomalies | Zero and extreme values must be distinguished from missingness | Retain field, add missing-indicator feature, impute with a reproducible strategy, and evaluate capped variant | Provisional |
| `NumberOfOpenCreditLinesAndLoans` | Count feature | Yes | No current missingness treatment required | Large values should be monitored but are not currently treated as a schema or warning failure by default | Retain field and assess whether clipping improves stability | Provisional |
| `NumberOfTimes90DaysLate` | Delinquency count feature | Yes | No current missingness treatment required | Values above `20` are suspicious and must be flagged; identical extreme pattern to other delinquency count fields | Retain field, create capped version, and assess whether this field should also support a severe-history indicator | Provisional |
| `NumberRealEstateLoansOrLines` | Count feature | Yes | No current missingness treatment required | Large values should be monitored but are not currently treated as a hard issue | Retain field and review distribution during Stage 3 feature stability checks | Provisional |
| `NumberOfTime60-89DaysPastDueNotWorse` | Delinquency count feature | Yes | No current missingness treatment required | Values above `20` are suspicious and must be flagged; identical extreme pattern to the other delinquency count fields | Retain field, create capped version, and assess interaction with the other delinquency counts | Provisional |
| `NumberOfDependents` | Count feature | Yes | Missing values require explicit imputation | Negative values are invalid if encountered; high-end counts should be monitored | Retain field, add missing-indicator feature, impute reproducibly, and review whether capping is needed | Provisional |

## Stage 3 Input Contract

Stage 3 should assume the following:

- The feature pipeline consumes the columns listed above as in-scope predictive inputs, except `Unnamed: 0` and `SeriousDlqin2yrs`.
- Missingness logic must be implemented for `MonthlyIncome` and `NumberOfDependents`.
- Warning-level anomaly handling must be implemented for `age`, `DebtRatio`, `RevolvingUtilizationOfUnsecuredLines`, and the three delinquency count fields.
- The first baseline pipeline should preserve enough transparency to compare raw versus bounded feature variants.

## Explicit Non-Decisions

The following details are not yet locked and should be finalized during Stage 3 implementation experiments:

- the exact imputation statistic for `MonthlyIncome`
- the exact imputation statistic for `NumberOfDependents`
- the final clipping thresholds used inside the feature pipeline
- whether transformed numeric variants replace raw values or coexist with them
- whether delinquency features should receive additional bucketed or binary-derived forms

These are implementation details, not reasons to reopen the field-level scope decision.

## Stage 2 Gate Relevance

This document is part of the Stage 2 gate before Stage 3. Stage 2 should not be considered closed until:

- every source field has a recorded treatment decision
- validation rules reflect the agreed warning-level issues
- Stage 3 can start without re-debating basic source-field treatment scope
