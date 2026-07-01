from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_TORCH_SCRIPT = PROJECT_ROOT / "scripts" / "train_torch.py"


def _load_train_torch_module():
    spec = importlib.util.spec_from_file_location(
        "train_torch_script",
        TRAIN_TORCH_SCRIPT,
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
            model_name="torch_mlp_challenger",
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


def _build_training_dataframe(row_count: int = 36) -> pd.DataFrame:
    rows: list[dict[str, float | int | None]] = []
    for index in range(row_count):
        is_positive = index % 4 == 0
        rows.append(
            {
                "Unnamed: 0": index + 1,
                "SeriousDlqin2yrs": 1 if is_positive else 0,
                "RevolvingUtilizationOfUnsecuredLines": 0.82 if is_positive else 0.10 + index * 0.004,
                "age": 30 if is_positive else 45 + (index % 15),
                "NumberOfTime30-59DaysPastDueNotWorse": 3 if is_positive else 0,
                "DebtRatio": 1.20 if is_positive else 0.18 + index * 0.01,
                "MonthlyIncome": None if index in {0, 8, 20} else (2600 if is_positive else 5400 + index * 30),
                "NumberOfOpenCreditLinesAndLoans": 3 if is_positive else 8 + (index % 4),
                "NumberOfTimes90DaysLate": 2 if is_positive else 0,
                "NumberRealEstateLoansOrLines": 0 if is_positive else 1 + (index % 2),
                "NumberOfTime60-89DaysPastDueNotWorse": 1 if is_positive else 0,
                "NumberOfDependents": None if index in {0, 9} else (2 if is_positive else index % 2),
            }
        )
    return pd.DataFrame(rows)


def _write_config(path: Path) -> None:
    lines = [
        "metadata:",
        "  model_name: torch_mlp_challenger",
        "  model_family: pytorch_mlp",
        "  model_role: challenger",
        "  model_version: 0.2.0",
        "  schema_version: 1.1.0",
        "runtime:",
        "  device: cpu",
        "  random_state: 42",
        "training:",
        "  hidden_dims: [12, 6]",
        "  dropout: 0.05",
        "  learning_rate: 0.01",
        "  batch_size: 8",
        "  max_epochs: 6",
        "  patience: 3",
        "  min_delta: 0.0001",
        "  weight_decay: 0.0",
        "  positive_class_weight_strategy: auto_from_train_ratio",
        "  positive_class_weight_value: null",
        "  threshold: 0.4",
        "artifacts:",
        "  output_dir: artifacts/torch",
        "  model_file: torch_mlp_challenger.pt",
        "  config_snapshot_file: torch_config_snapshot.json",
        "  feature_schema_file: feature_schema.json",
        "  split_metadata_file: split_metadata.json",
        "  metrics_file: torch_metrics.json",
        "  run_summary_file: run_summary.json",
        "  training_history_file: training_history.json",
        "guardrails:",
        "  require_finite_probabilities: true",
        "  require_complete_metadata: true",
        "  require_reload_check: true",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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


def test_train_torch_main_writes_expected_artifacts(monkeypatch, tmp_path: Path) -> None:
    module = _load_train_torch_module()
    output_dir = tmp_path / "artifacts" / "torch"
    config_path = tmp_path / "model_torch.yaml"
    _write_config(config_path)
    training_data = _build_training_dataframe()

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            config=config_path,
            output_dir=output_dir,
            threshold=0.4,
            model_version=None,
            schema_version=None,
            verbose=False,
        ),
    )
    monkeypatch.setattr(module, "get_settings", lambda: _build_settings(tmp_path))
    monkeypatch.setattr(module, "build_repository", lambda settings: FakeRepository(training_data))

    exit_code = module.main()

    assert exit_code == 0
    assert (output_dir / "torch_mlp_challenger.pt").exists()
    assert (output_dir / "torch_config_snapshot.json").exists()
    assert (output_dir / "feature_schema.json").exists()
    assert (output_dir / "split_metadata.json").exists()
    assert (output_dir / "torch_metrics.json").exists()
    assert (output_dir / "training_history.json").exists()
    assert (output_dir / "run_summary.json").exists()

    metrics_payload = json.loads((output_dir / "torch_metrics.json").read_text())
    split_metadata_payload = json.loads((output_dir / "split_metadata.json").read_text())
    history_payload = json.loads((output_dir / "training_history.json").read_text())
    run_summary_payload = json.loads((output_dir / "run_summary.json").read_text())

    assert metrics_payload["model_name"] == "torch_mlp_challenger"
    assert metrics_payload["model_version"] == "0.2.0"
    assert metrics_payload["schema_version"] == "1.1.0"
    assert metrics_payload["threshold"] == 0.4
    assert metrics_payload["dataset_fingerprint"] == "fixture-fingerprint"
    assert metrics_payload["partitions"]["validation"]["average_precision"] >= 0.0
    assert metrics_payload["partitions"]["validation"]["roc_auc"] >= 0.0

    assert split_metadata_payload["dataset_fingerprint"] == "fixture-fingerprint"
    assert split_metadata_payload["partitions"]["train"]["row_count"] > 0
    assert split_metadata_payload["partitions"]["validation"]["row_count"] > 0
    assert split_metadata_payload["partitions"]["test"]["row_count"] > 0

    assert history_payload["model_name"] == "torch_mlp_challenger"
    assert history_payload["history"]["epochs_completed"] >= 1
    assert len(history_payload["history"]["history"]) == history_payload["history"]["epochs_completed"]

    assert run_summary_payload["model_name"] == "torch_mlp_challenger"
    assert run_summary_payload["reload_check"]["status"] == "passed"
    assert run_summary_payload["reload_check"]["row_count"] == split_metadata_payload["partitions"]["validation"]["row_count"]


def test_train_torch_main_returns_error_when_validation_fails(monkeypatch, tmp_path: Path) -> None:
    module = _load_train_torch_module()
    output_dir = tmp_path / "artifacts" / "torch"
    config_path = tmp_path / "model_torch.yaml"
    _write_config(config_path)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            config=config_path,
            output_dir=output_dir,
            threshold=None,
            model_version=None,
            schema_version=None,
            verbose=False,
        ),
    )
    monkeypatch.setattr(module, "get_settings", lambda: _build_settings(tmp_path))
    monkeypatch.setattr(
        module,
        "build_repository",
        lambda settings: FakeRepository(
            _build_training_dataframe(),
            validation_errors=["training file failed schema validation"],
        ),
    )

    exit_code = module.main()

    assert exit_code == 1
    assert not (output_dir / "torch_mlp_challenger.pt").exists()
