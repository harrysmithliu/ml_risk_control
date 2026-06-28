# Data Dictionary

## Overview

This project uses Kaggle's *Give Me Some Credit* dataset as the primary public benchmark for credit risk modeling. The training target is `SeriousDlqin2yrs`, which indicates whether a borrower experienced 90 days past due or worse within two years.

This document describes the raw files currently used in local development, the expected field meanings, the latest validation snapshot, and the main dataset limitations that must remain visible throughout the project.

## Source and Access

- Source: Kaggle competition dataset, *Give Me Some Credit*
- Local raw data directory: `data/raw/GiveMeSomeCredit`
- Raw data is intentionally excluded from Git
- Validation report path: `artifacts/validation/raw_data_validation_report.json`
- Latest dataset fingerprint: `61f2f7b456e4481a4e84fba49adf22cd9d6915632df5e64df3374301b961c29d`

## Raw Files

| File | Role | Rows | Notes |
|---|---|---:|---|
| `cs-training.csv` | Labeled training dataset | 150,000 | Contains the binary target column |
| `cs-test.csv` | Unlabeled benchmark dataset | 101,503 | Includes an empty `SeriousDlqin2yrs` placeholder column in the local copy |
| `sampleEntry.csv` | Sample submission template | 101,503 | Contains `Id` and `Probability` only |
| `Data Dictionary.xls` | Source-provided field reference | N/A | Supplementary source artifact |

## Field Dictionary

| Column | Type | Present In | Description | Notes |
|---|---|---|---|---|
| `Unnamed: 0` | Integer | Train, Test | Source row identifier | Treated as an ID column, not a predictive feature |
| `SeriousDlqin2yrs` | Binary integer | Train | Target flag for serious delinquency within two years | Positive class is `1` |
| `RevolvingUtilizationOfUnsecuredLines` | Numeric ratio | Train, Test | Total balance on credit cards and personal lines of credit, excluding real estate and installment debt, divided by total credit limits | May contain values greater than 1 |
| `age` | Integer | Train, Test | Borrower age in years | Must be reviewed for implausible values during EDA |
| `NumberOfTime30-59DaysPastDueNotWorse` | Integer count | Train, Test | Number of times the borrower was 30 to 59 days past due, not worse, in the observed history | Delinquency count feature |
| `DebtRatio` | Numeric ratio | Train, Test | Monthly debt payments, alimony, and living costs divided by gross monthly income | May include extreme outliers |
| `MonthlyIncome` | Numeric | Train, Test | Reported monthly income | Contains substantial missingness |
| `NumberOfOpenCreditLinesAndLoans` | Integer count | Train, Test | Number of open loans such as installment loans, real estate loans, and credit cards | Exposure breadth feature |
| `NumberOfTimes90DaysLate` | Integer count | Train, Test | Number of times the borrower was 90 days or more past due | Strong delinquency severity signal candidate |
| `NumberRealEstateLoansOrLines` | Integer count | Train, Test | Number of mortgage and real estate related loans or credit lines | Exposure mix feature |
| `NumberOfTime60-89DaysPastDueNotWorse` | Integer count | Train, Test | Number of times the borrower was 60 to 89 days past due, not worse, in the observed history | Delinquency count feature |
| `NumberOfDependents` | Numeric count | Train, Test | Number of dependents in the household | Contains some missingness |

## Validation Snapshot

### Training File

- File status: passed
- Row count: `150,000`
- Column count: `12`
- ID uniqueness: passed
- Target binary check: passed
- Positive class count: `10,026`
- Negative class count: `139,974`
- Positive class rate: `6.684%`

### Test File

- File status: passed
- Row count: `101,503`
- Column count: `12`
- ID uniqueness: passed
- Validation warning: `SeriousDlqin2yrs` is present as an empty placeholder column

### Missingness Snapshot

| Column | Train Missing | Train Missing % | Test Missing | Test Missing % |
|---|---:|---:|---:|---:|
| `MonthlyIncome` | 29,731 | 19.82% | 20,103 | 19.81% |
| `NumberOfDependents` | 3,924 | 2.62% | 2,626 | 2.59% |

All other validated columns are currently complete in both the training and test files.

## Modeling Notes

- `Unnamed: 0` must be excluded from model features.
- `SeriousDlqin2yrs` is available only for training and evaluation workflows.
- `cs-test.csv` must be treated as unlabeled even if an empty target placeholder column is present.
- Missingness in `MonthlyIncome` and `NumberOfDependents` must be handled explicitly in preprocessing.
- Because the target is imbalanced, downstream evaluation must prioritize PR-AUC, ROC-AUC, threshold analysis, and class-sensitive metrics over accuracy alone.

## Known Dataset Limitations

- The dataset is anonymized and cannot support business interpretation at the level of real production attributes.
- No reliable event-time field is available for true out-of-time validation.
- The benchmark dataset is useful for prototyping, but it is not evidence of production-grade policy performance.
- Feature explanations derived from this dataset describe model behavior, not causality.
- The local `cs-test.csv` file includes an empty target placeholder column, which must not be treated as labeled ground truth.
