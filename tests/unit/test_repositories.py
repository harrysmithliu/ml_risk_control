from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_risk_control.config import (
    AppSettings,
    ArtifactSettings,
    BackendType,
    DataSettings,
    FeatureFlags,
    Settings,
    SnowflakeSettings,
    StreamlitSettings,
    TrainingSettings,
)
from ml_risk_control.data.repositories import (
    LocalRepository,
    RepositoryValidationError,
    SnowflakeRepository,
    build_repository,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_train_rows() -> list[dict[str, object]]:
    return [
        {
            "Unnamed: 0": 1,
            "SeriousDlqin2yrs": 0,
            "RevolvingUtilizationOfUnsecuredLines": 0.15,
            "age": 45,
            "NumberOfTime30-59DaysPastDueNotWorse": 0,
            "DebtRatio": 0.32,
            "MonthlyIncome": 5000,
            "NumberOfOpenCreditLinesAndLoans": 8,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60-89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 2,
        },
        {
            "Unnamed: 0": 2,
            "SeriousDlqin2yrs": 1,
            "RevolvingUtilizationOfUnsecuredLines": 0.91,
            "age": 38,
            "NumberOfTime30-59DaysPastDueNotWorse": 1,
            "DebtRatio": 0.67,
            "MonthlyIncome": 3200,
            "NumberOfOpenCreditLinesAndLoans": 5,
            "NumberOfTimes90DaysLate": 1,
            "NumberRealEstateLoansOrLines": 0,
            "NumberOfTime60-89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 1,
        },
    ]


def _build_test_rows() -> list[dict[str, object]]:
    return [
        {
            "Unnamed: 0": 3,
            "SeriousDlqin2yrs": None,
            "RevolvingUtilizationOfUnsecuredLines": 0.40,
            "age": 41,
            "NumberOfTime30-59DaysPastDueNotWorse": 0,
            "DebtRatio": 0.25,
            "MonthlyIncome": 6100,
            "NumberOfOpenCreditLinesAndLoans": 7,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60-89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 0,
        }
    ]


def _build_sample_submission_rows() -> list[dict[str, object]]:
    return [{"Id": 3, "Probability": 0.12}]


def _build_settings(raw_data_dir: Path, *, backend: BackendType = "local") -> Settings:
    return Settings(
        project_root=raw_data_dir.parent.parent,
        app=AppSettings(environment="test", debug=False),
        data=DataSettings(
            backend=backend,
            raw_data_dir=raw_data_dir,
            interim_data_dir=raw_data_dir.parent.parent / "data" / "interim",
            processed_data_dir=raw_data_dir.parent.parent / "data" / "processed",
            train_file="cs-training.csv",
            test_file="cs-test.csv",
            data_dictionary_file="Data Dictionary.xls",
        ),
        artifacts=ArtifactSettings(
            artifact_dir=raw_data_dir.parent.parent / "artifacts",
            model_dir=raw_data_dir.parent.parent / "artifacts" / "models",
            report_dir=raw_data_dir.parent.parent / "reports" / "figures",
            model_name="xgboost_credit_risk",
            model_version="0.1.0",
        ),
        training=TrainingSettings(
            target_column="SeriousDlqin2yrs",
            id_column="Unnamed: 0",
            random_state=42,
            test_size=0.2,
            cv_folds=5,
            positive_class_label=1,
            decision_threshold=0.5,
        ),
        streamlit=StreamlitSettings(server_port=8501, server_address="0.0.0.0"),
        snowflake=SnowflakeSettings(
            account="",
            user="",
            password="",
            warehouse="",
            database="ML_RISK_CONTROL",
            schema="RAW",
            role="",
            raw_schema="RAW",
            curated_schema="CURATED",
            features_schema="FEATURES",
            serving_schema="SERVING",
        ),
        features=FeatureFlags(
            enable_snowflake_writeback=False,
            enable_monitoring_export=False,
        ),
    )


def test_build_repository_returns_local_repository_for_local_backend(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path, backend="local")

    repository = build_repository(settings)

    assert isinstance(repository, LocalRepository)


def test_build_repository_returns_snowflake_repository_for_snowflake_backend(
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path, backend="snowflake")

    repository = build_repository(settings)

    assert isinstance(repository, SnowflakeRepository)


def test_local_repository_loads_training_and_scoring_data(tmp_path: Path) -> None:
    _write_csv(tmp_path / "cs-training.csv", _build_train_rows())
    _write_csv(tmp_path / "cs-test.csv", _build_test_rows())
    settings = _build_settings(tmp_path)
    repository = LocalRepository(settings)

    training_data = repository.load_training_data()
    scoring_data = repository.load_scoring_data()

    assert training_data.shape == (2, 12)
    assert scoring_data.shape == (1, 12)
    assert training_data["SeriousDlqin2yrs"].tolist() == [0, 1]


def test_local_repository_returns_none_when_sample_submission_is_missing(tmp_path: Path) -> None:
    _write_csv(tmp_path / "cs-training.csv", _build_train_rows())
    _write_csv(tmp_path / "cs-test.csv", _build_test_rows())
    settings = _build_settings(tmp_path)
    repository = LocalRepository(settings)

    sample_submission = repository.load_sample_submission_data()

    assert sample_submission is None


def test_local_repository_loads_sample_submission_when_present(tmp_path: Path) -> None:
    _write_csv(tmp_path / "cs-training.csv", _build_train_rows())
    _write_csv(tmp_path / "cs-test.csv", _build_test_rows())
    _write_csv(tmp_path / "sampleEntry.csv", _build_sample_submission_rows())
    settings = _build_settings(tmp_path)
    repository = LocalRepository(settings)

    sample_submission = repository.load_sample_submission_data()

    assert sample_submission is not None
    assert sample_submission.shape == (1, 2)
    assert sample_submission.columns.tolist() == ["Id", "Probability"]


def test_local_repository_raises_validation_error_for_invalid_training_file(
    tmp_path: Path,
) -> None:
    invalid_rows = _build_train_rows()
    for row in invalid_rows:
        row.pop("MonthlyIncome")

    _write_csv(tmp_path / "cs-training.csv", invalid_rows)
    _write_csv(tmp_path / "cs-test.csv", _build_test_rows())
    settings = _build_settings(tmp_path)
    repository = LocalRepository(settings)

    with pytest.raises(RepositoryValidationError, match="Required columns are missing"):
        repository.load_training_data()


def test_local_repository_validate_raw_data_returns_passed_report(tmp_path: Path) -> None:
    _write_csv(tmp_path / "cs-training.csv", _build_train_rows())
    _write_csv(tmp_path / "cs-test.csv", _build_test_rows())
    settings = _build_settings(tmp_path)
    repository = LocalRepository(settings)

    report = repository.validate_raw_data()

    assert report["status"] == "passed"
    assert report["backend"] == "local"
    assert report["files"]["train"]["errors"] == []
    assert report["files"]["sample_submission"]["warnings"] == ["File does not exist."]
