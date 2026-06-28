# Data Quality Rules

## Purpose

This document defines the current data-quality rule set for the Stage 2 foundation. It separates hard validation failures from warning-level findings so that the project can preserve strict schema integrity while still surfacing business-relevant anomalies discovered during EDA.

The rules in this document apply first to the raw local CSV workflow and are intended to remain compatible with the future Snowflake-backed path.

## Rule Tiers

### Tier 1 — Hard Failures

These rules stop the validation flow and mark the dataset or file as failed.

### Tier 2 — Warning-Level Findings

These rules do not fail validation by themselves, but they must be surfaced in validation output and considered in curated processing and feature engineering.

### Tier 3 — Planned Rules

These rules are already supported conceptually by Stage 2 decisions, but are not yet fully enforced in code as independent checks or escalated thresholds.

## Current Hard Failure Rules

The following rules are currently implemented in `src/ml_risk_control/data/validation.py` as validation errors:

| Rule Area | Condition | Scope | Current Outcome |
|---|---|---|---|
| File existence | Required file is missing | Train / Test | Validation fails |
| Schema completeness | Required columns are missing | Train / Test / Sample submission | Validation fails |
| Unexpected columns | Non-optional columns appear outside the contract | Train / Test / Sample submission | Validation fails |
| Duplicate columns | Duplicate column names are present | All validated files | Validation fails |
| Identifier presence | Declared identifier column is missing | Train / Test | Validation fails |
| Target presence | Required target column is missing | Train | Validation fails |
| Target integrity | Training target contains values outside `{0, 1}` | Train | Validation fails |
| Unlabeled-file target misuse | Test target placeholder contains non-null values | Test | Validation fails |
| Empty file | File contains zero rows | All validated files | Validation fails |

## Current Warning-Level Rules

The following warning-level rules are currently implemented and emitted through `data_quality_flags` and human-readable warning messages:

| Rule Area | Condition | Output Type | Current Threshold |
|---|---|---|---|
| Identifier uniqueness | Identifier column is not unique | Warning | Any duplicate ID |
| Test target placeholder | Optional empty target column is present in `cs-test.csv` | Warning | Presence of empty placeholder target |
| Age anomaly | Borrower age below 18 | Warning | `< 18` |
| Age anomaly | Borrower age above 100 | Warning | `> 100` |
| Revolving utilization anomaly | Utilization exceeds normal ratio range | Warning | `> 1.0` |
| Debt ratio anomaly | Debt ratio is unusually large | Warning | `> 5.0` |
| Monthly income anomaly | Non-missing monthly income is non-positive | Warning | `<= 0` |
| Dependent count anomaly | Non-missing dependent count is negative | Warning | `< 0` |
| Delinquency anomaly | Delinquency count is unusually large | Warning | `> 20` |

## Structured Output Contract

Current validation results expose the warning-level findings in two ways:

1. `warnings`
   - human-readable summary messages
2. `data_quality_flags`
   - machine-consumable counts for each warning family

This separation should be preserved. Downstream code should rely on `data_quality_flags` for logic and use `warnings` for user-facing diagnostics.

## Current Implemented Thresholds

The current warning thresholds are intentionally conservative and EDA-driven:

| Field or Field Group | Current Threshold | Reasoning |
|---|---|---|
| `age` | `< 18` or `> 100` | Rare, likely implausible demographic values |
| `RevolvingUtilizationOfUnsecuredLines` | `> 1.0` | Useful first-pass flag for over-limit or unusual ratio behavior |
| `DebtRatio` | `> 5.0` | Practical anomaly marker for severe scale distortion |
| `MonthlyIncome` | `<= 0` for non-missing rows | Zero or negative income should be reviewed separately from missingness |
| `NumberOfDependents` | `< 0` for non-missing rows | Negative household dependents are invalid |
| Delinquency count columns | `> 20` | Strong signal of suspiciously large repayment-history counts in this benchmark |

## Rule Ownership by Stage

### Stage 1 Ownership

Already completed:

- file contract definition
- schema validation
- raw validation report generation
- warning-capable validation output

### Stage 2 Ownership

Must be completed before Stage 3 begins:

- finalize which warning-level rules remain warnings versus become curated-layer hard filters
- document field-level treatment responses for each flagged condition
- align repository and curated processing behavior with these rules

### Stage 3 Ownership

Will consume, not redefine:

- capped or transformed feature variants
- missing-value imputation implementation
- optional anomaly indicator features

## Planned but Not Yet Fully Formalized Rules

The following directions are already justified by EDA and treatment decisions, but are not yet fully encoded as separate rule objects or curated-stage enforcement:

| Area | Planned Direction |
|---|---|
| Missingness thresholds | Missingness should be explicitly summarized and tracked at the column level, especially for `MonthlyIncome` and `NumberOfDependents` |
| Curated row handling | Some anomaly rows may be retained with warning flags rather than immediately dropped |
| Clipping policy | Long-tail numeric fields will likely require reproducible clipping or bounded variants in the feature pipeline |
| Multi-column delinquency logic | The shared extreme-value pattern across the three delinquency columns should be reviewed jointly, not only one column at a time |
| Curated-layer audit logging | Future curated processing should record how many rows were capped, imputed, or otherwise adjusted |

## Recommended Rule Escalation Policy

The project should use the following escalation logic moving forward:

- Keep schema and target-integrity rules as hard failures.
- Keep low-frequency but plausible business anomalies as warnings unless they clearly break downstream modeling assumptions.
- Escalate a warning to a hard curated-stage rule only when both are true:
  - the anomaly is demonstrably invalid or destructive for modeling
  - the treatment decision is stable enough to be enforced reproducibly

## Stage 2 Gate Relevance

This document helps define the Stage 2 gate before Stage 3. Stage 2 should not be considered closed until:

- the rule tiers are clearly documented
- warning-level findings are consistent with EDA conclusions
- field treatment decisions explain how each warning family will be handled later
- Stage 3 can implement preprocessing without re-arguing the data-quality rule scope
