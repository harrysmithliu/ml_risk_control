-- Credit Risk Intelligence Platform
-- Stage 2 Snowflake foundation
-- Purpose: create source-aligned RAW tables for the Give Me Some Credit dataset.

USE DATABASE ML_RISK_CONTROL;
USE SCHEMA RAW;

CREATE TABLE IF NOT EXISTS GMSC_TRAIN (
    RAW_INGESTION_ID VARCHAR NOT NULL,
    SOURCE_FILE_NAME VARCHAR NOT NULL,
    SOURCE_ROW_NUMBER NUMBER(38,0) NOT NULL,
    DATASET_FINGERPRINT VARCHAR,
    INGESTED_AT_UTC TIMESTAMP_TZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    "Unnamed: 0" NUMBER(38,0),
    "SeriousDlqin2yrs" NUMBER(1,0),
    "RevolvingUtilizationOfUnsecuredLines" FLOAT,
    "age" NUMBER(38,0),
    "NumberOfTime30-59DaysPastDueNotWorse" NUMBER(38,0),
    "DebtRatio" FLOAT,
    "MonthlyIncome" FLOAT,
    "NumberOfOpenCreditLinesAndLoans" NUMBER(38,0),
    "NumberOfTimes90DaysLate" NUMBER(38,0),
    "NumberRealEstateLoansOrLines" NUMBER(38,0),
    "NumberOfTime60-89DaysPastDueNotWorse" NUMBER(38,0),
    "NumberOfDependents" FLOAT
)
COMMENT = 'Raw Kaggle training rows aligned to cs-training.csv with ingestion metadata.';

CREATE TABLE IF NOT EXISTS GMSC_TEST (
    RAW_INGESTION_ID VARCHAR NOT NULL,
    SOURCE_FILE_NAME VARCHAR NOT NULL,
    SOURCE_ROW_NUMBER NUMBER(38,0) NOT NULL,
    DATASET_FINGERPRINT VARCHAR,
    INGESTED_AT_UTC TIMESTAMP_TZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    "Unnamed: 0" NUMBER(38,0),
    "SeriousDlqin2yrs" NUMBER(1,0),
    "RevolvingUtilizationOfUnsecuredLines" FLOAT,
    "age" NUMBER(38,0),
    "NumberOfTime30-59DaysPastDueNotWorse" NUMBER(38,0),
    "DebtRatio" FLOAT,
    "MonthlyIncome" FLOAT,
    "NumberOfOpenCreditLinesAndLoans" NUMBER(38,0),
    "NumberOfTimes90DaysLate" NUMBER(38,0),
    "NumberRealEstateLoansOrLines" NUMBER(38,0),
    "NumberOfTime60-89DaysPastDueNotWorse" NUMBER(38,0),
    "NumberOfDependents" FLOAT
)
COMMENT = 'Raw Kaggle test rows aligned to cs-test.csv; target placeholder remains nullable and unlabeled.';

CREATE TABLE IF NOT EXISTS GMSC_SAMPLE_SUBMISSION (
    RAW_INGESTION_ID VARCHAR NOT NULL,
    SOURCE_FILE_NAME VARCHAR NOT NULL,
    SOURCE_ROW_NUMBER NUMBER(38,0) NOT NULL,
    DATASET_FINGERPRINT VARCHAR,
    INGESTED_AT_UTC TIMESTAMP_TZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    "Id" NUMBER(38,0),
    "Probability" FLOAT
)
COMMENT = 'Raw Kaggle sample submission rows aligned to sampleEntry.csv with ingestion metadata.';
