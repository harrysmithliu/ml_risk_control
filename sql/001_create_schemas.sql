-- Credit Risk Intelligence Platform
-- Stage 2 Snowflake foundation
-- Purpose: create the logical schema layers used by the project.

CREATE DATABASE IF NOT EXISTS ML_RISK_CONTROL;

USE DATABASE ML_RISK_CONTROL;

CREATE SCHEMA IF NOT EXISTS RAW
    COMMENT = 'Immutable source-aligned records and ingestion metadata.';

CREATE SCHEMA IF NOT EXISTS CURATED
    COMMENT = 'Validated and cleaned credit risk records.';

CREATE SCHEMA IF NOT EXISTS FEATURES
    COMMENT = 'Model-ready feature snapshots and feature schema outputs.';

CREATE SCHEMA IF NOT EXISTS SERVING
    COMMENT = 'Predictions, model versions, monitoring inputs, and serving metadata.';
