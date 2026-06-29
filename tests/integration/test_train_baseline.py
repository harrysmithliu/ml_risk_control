from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ml_risk_control.config import (
    AppSettings,
    ArtifactSettings,
    DataSettings,
    FeatureFlags,
    Settings,
    SnowflakeSettings,
    StreamlitSettings,
    TrainingSettings,
)
from ml_risk_control.data.repositories import RepositoryValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_BASELINE_SCRIPT = PROJECT_ROOT / "scripts" / "train_baseline.py"


def _load_train_baseline_module():
    spec = importlib.util.spec_from_file_location(
        "train_baseline_script",
        TRAIN_BASELINE_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_settings(project_root: Path) -> Settings:
    raw_data_dir = project_root / "data" / "raw"
    return Settings(
        project_root=project_root,
        app=AppSettings(environment="test", debug=False),
        data=DataSettings(
            backend="local",
            raw_data_dir=raw_data_dir,
            interim_data_dir=project_root / "data" / "interim",
            processed_data_dir=project_root / "data" / "processed",
            train_file="cs-training.csv",
            test_file="cs-test.csv",
            data_dictionary_file="Data Dictionary.xls",
        ),
        artifacts=ArtifactSettings(
            artifact_dir=project_root / "artifacts",
            model_dir=project_root / "artifacts" / "models",
            report_dir=project_root / "reports" / "figures",
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


def _build_training_dataframe(row_count: int = 9) -> pd.DataFrame:
    rows: list[dict[str, float | int | None]] = []
    for index in range(row_count):
        rows.append(
            {
                "Unnamed: 0": index + 1,
                "SeriousDlqin2yrs": 1 if index % 3 == 0 else 0,
                "RevolvingUtilizationOfUnsecuredLines": 0.10 + index * 0.01,
                "age": 30 + index,
                "NumberOfTime30-59DaysPastDueNotWorse": index % 2,
                "DebtRatio": 0.20 + index * 0.01,
                "MonthlyIncome": 3000 + index * 100,
                "NumberOfOpenCreditLinesAndLoans": 4 + (index % 4),
                "NumberOfTimes90DaysLate": index % 2,
                "NumberRealEstateLoansOrLines": index % 3,
                "NumberOfTime60-89DaysPastDueNotWorse": index % 2,
                "NumberOfDependents": index % 3,
            }
        )
    return pd.DataFrame(rows)


class FakeRepository:
    backend = "local"

    def __init__(self, training_data: pd.DataFrame, *, validation_errors: list[str] | None = None):
        self._training_data = training_data
        self._validation_errors = validation_errors or []

    def validate_raw_data(self) -> dict[str, object]:
        return {
            "status": "failed" if self._validation_errors else "passed",
            "backend": self.backend,
            "dataset_fingerprint": "fixture-fingerprint",
            "files": {
                "train": {"errors": self._validation_errors, "warnings": []},
                "test": {"errors": [], "warnings": []},
                "sample_submission": {"errors": [], "warnings": []},
            },
        }

    def load_training_data(self) -> pd.DataFrame:
        return self._training_data.copy()


class FakeSplitConfig:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state


class FakeLogisticRegressionBaselineConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeFeatureSchema:
    def __init__(self, schema_version: str):
        self.schema_version = schema_version

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "raw_feature_columns": ["MonthlyIncome", "DebtRatio"],
            "model_input_features": [
                "MonthlyIncome",
                "DebtRatio",
                "MonthlyIncome_missing",
            ],
        }


class FakeLogisticRegressionBaseline:
    def __init__(
        self,
        *,
        id_column: str,
        target_column: str,
        model_version: str,
        schema_version: str,
        config: FakeLogisticRegressionBaselineConfig,
    ) -> None:
        self.id_column = id_column
        self.target_column = target_column
        self.model_version = model_version
        self.schema_version = schema_version
        self.config = config
        self.feature_schema = FakeFeatureSchema(schema_version)
        self.training_summary_ = None

    def fit(self, dataframe: pd.DataFrame) -> FakeLogisticRegressionBaseline:
        self.training_summary_ = {
            "row_count": int(len(dataframe)),
            "positive_rate": float((dataframe[self.target_column] == 1).mean()),
        }
        return self

    def predict_proba(self, dataframe: pd.DataFrame) -> pd.Series:
        values = [0.8 if value == 1 else 0.2 for value in dataframe[self.target_column].tolist()]
        return pd.Series(values, index=dataframe.index, name="predicted_probability")

    def build_artifact_metadata(self) -> dict[str, object]:
        return {
            "model_name": "logistic_regression_baseline",
            "model_version": self.model_version,
            "schema_version": self.schema_version,
            "pipeline_steps": ["preprocessing", "classifier"],
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake baseline artifact", encoding="utf-8")
        return path


def _fake_split_training_data(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    config: FakeSplitConfig,
):
    del target_column, config
    return SimpleNamespace(
        train=dataframe.iloc[:5].reset_index(drop=True),
        validation=dataframe.iloc[5:7].reset_index(drop=True),
        test=dataframe.iloc[7:9].reset_index(drop=True),
    )


def _fake_build_split_metadata(partitions, *, target_column: str, config: FakeSplitConfig):
    del config
    return {
        "config": {"random_state": 42},
        "partitions": {
            "train": {"row_count": int(len(partitions.train))},
            "validation": {"row_count": int(len(partitions.validation))},
            "test": {"row_count": int(len(partitions.test))},
        },
        "target_column": target_column,
    }


def _fake_evaluate_binary_classifier(y_true, y_score, *, threshold: float):
    return {
        "row_count": len(y_true),
        "positive_rate": float((pd.Series(y_true) == 1).mean()),
        "threshold": threshold,
        "average_precision": 0.75,
        "roc_auc": 0.80,
        "ks_statistic": 0.60,
        "brier_score": 0.18,
        "accuracy": 0.70,
        "precision": 0.67,
        "recall": 0.75,
        "f1": 0.71,
        "confusion_matrix": {
            "counts": {"matrix": [[1, 0], [0, 1]]},
        },
        "mean_score": float(pd.Series(y_score).mean()),
    }


def test_train_baseline_main_writes_expected_artifacts(monkeypatch, tmp_path: Path) -> None:
    module = _load_train_baseline_module()
    output_dir = tmp_path / "artifacts" / "baseline"
    training_data = _build_training_dataframe()

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            output_dir=output_dir,
            threshold=0.4,
            model_version="0.2.0",
            schema_version="1.1.0",
            class_weight="balanced",
        ),
    )
    monkeypatch.setattr(module, "get_settings", lambda: _build_settings(tmp_path))
    monkeypatch.setattr(module, "build_repository", lambda settings: FakeRepository(training_data))
    monkeypatch.setattr(
        module,
        "_load_training_modules",
        lambda: (
            _fake_evaluate_binary_classifier,
            FakeSplitConfig,
            _fake_build_split_metadata,
            _fake_split_training_data,
            FakeLogisticRegressionBaseline,
            FakeLogisticRegressionBaselineConfig,
        ),
    )

    result = module.main()

    assert result == 0
    assert (output_dir / "logistic_regression_baseline.joblib").exists()
    assert (output_dir / "feature_schema.json").exists()
    assert (output_dir / "split_metadata.json").exists()
    assert (output_dir / "baseline_metrics.json").exists()
    assert (output_dir / "run_summary.json").exists()

    feature_schema_payload = json.loads((output_dir / "feature_schema.json").read_text())
    split_metadata_payload = json.loads((output_dir / "split_metadata.json").read_text())
    metrics_payload = json.loads((output_dir / "baseline_metrics.json").read_text())
    run_summary_payload = json.loads((output_dir / "run_summary.json").read_text())

    assert feature_schema_payload["feature_schema"]["schema_version"] == "1.1.0"
    assert split_metadata_payload["dataset_fingerprint"] == "fixture-fingerprint"
    assert split_metadata_payload["partitions"]["train"]["row_count"] == 5
    assert metrics_payload["threshold"] == 0.4
    assert metrics_payload["partitions"]["validation"]["average_precision"] == 0.75
    assert metrics_payload["model_version"] == "0.2.0"
    assert run_summary_payload["artifact_metadata"]["schema_version"] == "1.1.0"
    assert run_summary_payload["training_summary"]["row_count"] == 5


def test_train_baseline_main_returns_error_when_repository_validation_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_train_baseline_module()
    output_dir = tmp_path / "artifacts" / "baseline"
    training_data = _build_training_dataframe()

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            output_dir=output_dir,
            threshold=0.5,
            model_version="0.1.0",
            schema_version="1.0.0",
            class_weight="none",
        ),
    )
    monkeypatch.setattr(module, "get_settings", lambda: _build_settings(tmp_path))
    monkeypatch.setattr(
        module,
        "build_repository",
        lambda settings: FakeRepository(training_data, validation_errors=["Missing train file."]),
    )
    monkeypatch.setattr(
        module,
        "_load_training_modules",
        lambda: (
            _fake_evaluate_binary_classifier,
            FakeSplitConfig,
            _fake_build_split_metadata,
            _fake_split_training_data,
            FakeLogisticRegressionBaseline,
            FakeLogisticRegressionBaselineConfig,
        ),
    )

    result = module.main()

    assert result == 1
    assert not (output_dir / "baseline_metrics.json").exists()


def test_train_baseline_main_returns_error_when_repository_load_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_train_baseline_module()
    output_dir = tmp_path / "artifacts" / "baseline"

    class FailingRepository(FakeRepository):
        def load_training_data(self) -> pd.DataFrame:
            raise RepositoryValidationError("Training contract failed.")

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            output_dir=output_dir,
            threshold=0.5,
            model_version="0.1.0",
            schema_version="1.0.0",
            class_weight="none",
        ),
    )
    monkeypatch.setattr(module, "get_settings", lambda: _build_settings(tmp_path))
    monkeypatch.setattr(
        module,
        "build_repository",
        lambda settings: FailingRepository(_build_training_dataframe()),
    )
    monkeypatch.setattr(
        module,
        "_load_training_modules",
        lambda: (
            _fake_evaluate_binary_classifier,
            FakeSplitConfig,
            _fake_build_split_metadata,
            _fake_split_training_data,
            FakeLogisticRegressionBaseline,
            FakeLogisticRegressionBaselineConfig,
        ),
    )

    result = module.main()

    assert result == 1
    assert not (output_dir / "run_summary.json").exists()
