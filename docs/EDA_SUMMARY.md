# EDA Summary

## Scope

This summary documents the first reproducible exploratory data analysis pass on the raw `cs-training.csv` dataset from Kaggle's *Give Me Some Credit* benchmark.

The underlying machine-readable output is stored in:

- `artifacts/eda/eda_summary.json`

At the time of this run, figure generation was skipped because `matplotlib` was not installed in the local environment. The statistical summary remains valid and reproducible.

## Dataset Snapshot

- Source file: `data/raw/GiveMeSomeCredit/cs-training.csv`
- Rows: `150,000`
- Columns: `12`
- Duplicate rows: `0`
- Target column: `SeriousDlqin2yrs`
- Positive class count: `10,026`
- Negative class count: `139,974`
- Positive class rate: `6.684%`

## Key Findings

### 1. The target is materially imbalanced

The positive event rate is only `6.684%`, which confirms that accuracy should not be used as the primary model-selection metric in later stages.

Implication:

- Stage 3 and beyond must prioritize PR-AUC, ROC-AUC, recall/precision trade-offs, threshold analysis, and class-sensitive evaluation.

### 2. Missingness is concentrated in two columns

Only two validated columns currently contain missing values:

| Column | Missing Count | Missing Rate |
|---|---:|---:|
| `MonthlyIncome` | 29,731 | 19.82% |
| `NumberOfDependents` | 3,924 | 2.62% |

Implication:

- Both columns require explicit treatment decisions before feature-pipeline implementation.
- Missingness should be treated as a modeling and data-quality concern, not as a casual fill-in step.

### 3. Several numeric fields have severe long-tail behavior

The strongest extreme-value signals appear in:

- `DebtRatio`
- `RevolvingUtilizationOfUnsecuredLines`
- `MonthlyIncome`
- the three delinquency count columns

Examples from the current summary:

- `DebtRatio`: median `0.3665`, p99 `4979.04`, max `329664`
- `RevolvingUtilizationOfUnsecuredLines`: median `0.1542`, p99 `1.0930`, max `50708`
- `MonthlyIncome`: median `5400`, p99 `25000`, max `3008750`

Implication:

- Raw numeric scale is not trustworthy as-is for direct downstream modeling.
- Stage 2 must define whether these columns are clipped, transformed, winsorized, bucketed, or otherwise controlled.

### 4. Delinquency count columns contain suspicious extreme values

The three delinquency-history fields show identical counts of very large values:

- `NumberOfTime30-59DaysPastDueNotWorse`
- `NumberOfTime60-89DaysPastDueNotWorse`
- `NumberOfTimes90DaysLate`

Observed suspicious-value counts:

- values above `20`: `269` rows in each of the three columns
- values above `50`: `269` rows in each of the three columns
- column max: `98` in all three columns

Implication:

- These are strong candidates for business-rule review rather than being accepted as ordinary observations.
- A Stage 2 data-quality rule should at least flag such rows for warning-level inspection.

### 5. Age contains a small number of implausible values

Observed age anomalies:

- age `<= 0`: `1`
- age `< 18`: `1`
- age `> 100`: `13`
- max age: `109`

Implication:

- `age` is mostly well-behaved but not perfectly clean.
- Stage 2 should define whether implausible ages are dropped, capped, or retained with warning logic.

### 6. The raw identifier should remain excluded from features

`Unnamed: 0` behaves as a source row identifier rather than a meaningful predictive variable.

Implication:

- It should remain excluded from feature engineering and model training.

## Early Treatment Direction

The current EDA suggests the following preliminary treatment directions for later confirmation:

| Field Area | Preliminary Direction |
|---|---|
| `Unnamed: 0` | Exclude from features |
| `SeriousDlqin2yrs` | Use only as target |
| `MonthlyIncome` | Explicit missing-value treatment required |
| `NumberOfDependents` | Explicit missing-value treatment required |
| `DebtRatio` | Review for clipping or transformation |
| `RevolvingUtilizationOfUnsecuredLines` | Review for clipping or transformation |
| Delinquency count columns | Add suspicious-value rules and treatment review |
| `age` | Add implausible-value rule and treatment review |

These are not yet final feature-treatment decisions. They are the working hypotheses that should be formalized in the next Stage 2 steps.

## Recommended Next Actions

The next Stage 2 actions should proceed in this order:

1. Convert the findings above into explicit data-quality rules.
2. Decide which findings are hard failures versus warnings.
3. Create a documented treatment decision for every source field.
4. Update validation logic and tests to reflect the agreed rules.

## Current Limitation of This EDA Pass

- The current pass is based on summary statistics and rule-oriented scans.
- Chart files were not generated in this run because the plotting dependency was unavailable locally.
- This document does not yet represent the final treatment policy for Stage 3 feature engineering.
