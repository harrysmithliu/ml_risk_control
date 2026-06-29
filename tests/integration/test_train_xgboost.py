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
TRAIN_XGBOOST_SCRIPT = PROJECT_ROOT / "scripts" / "train_xgboost.py"


def _load_train_xgboost_module():
    spec = importlib.util.spec_from_file_location(
        "train_xgboost_script",
        TRAIN_XGBOOST_SCRIPT,
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


def _build_training_dataframe(row_count: int = 12) -> pd.DataFrame:
    rows: list[dict[str, float | int | None]] = []
    for index in range(row_count):
        rows.append(
            {
                "Unnamed: 0": index + 1,
                "SeriousDlqin2yrs": 1 if index % 4 == 0 else 0,
                "RevolvingUtilizationOfUnsecuredLines": 0.08 + index * 0.01,
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


def _write_config(
    path: Path,
    *,
    include_tuning: bool = False,
    include_scale_pos_weight: bool = False,
) -> None:
    lines = [
        "metadata:",
        "  model_name: xgboost_credit_risk",
        "  model_version: 0.2.0",
        "  schema_version: 1.1.0",
        "runtime:",
        "  random_state: 42",
        "training:",
        "  objective: binary:logistic",
        "  eval_metric: [aucpr, auc, logloss]",
        "  early_stopping_rounds: 25",
        "  threshold: 0.4",
        "reference_run:",
        "  params:",
        "    max_depth: 4",
        "    learning_rate: 0.05",
        "    n_estimators: 200",
    ]
    if include_tuning:
        lines.extend(
            [
                "tuning:",
                "  enabled: true",
                "  strategy: randomized_search",
                "  n_iter: 2",
                "  score_direction: maximize",
                "  scoring:",
                "    primary: average_precision",
                "    secondary: [roc_auc, ks_statistic]",
                "  parameter_space:",
                "    max_depth:",
                "      values: [3, 5]",
                "    learning_rate:",
                "      values: [0.05]",
                "    n_estimators:",
                "      values: [200]",
            ]
        )
    if include_scale_pos_weight:
        lines.extend(
            [
                "experiments:",
                "  class_imbalance:",
                "    run_original_distribution: true",
                "    run_scale_pos_weight_variant: true",
                "    scale_pos_weight_strategy: auto_from_train_ratio",
                "    run_smote_variant: false",
            ]
        )
    lines.extend(
        [
            "artifacts:",
            "  output_dir: artifacts/xgboost",
            "  model_file: xgboost_credit_risk.joblib",
            "  config_snapshot_file: xgboost_config_snapshot.json",
            "  feature_schema_file: feature_schema.json",
            "  split_metadata_file: split_metadata.json",
            "  metrics_file: xgboost_metrics.json",
            "  tuning_results_file: tuning_results.json",
            "  run_summary_file: run_summary.json",
            "  evaluation_rows:",
            "    - train",
            "    - validation",
            "    - test",
            "diagnostics:",
            "  native_importance_types: [gain, weight, cover]",
            "  save_learning_curve: true",
            "  learning_curve_file: learning_curve.json",
            "  save_curves: true",
            "  curves_file: curves.json",
            "  save_feature_importance_report: true",
            "  feature_importance_file: native_feature_importance.json",
            "  save_permutation_importance: true",
            "  permutation_importance_file: permutation_importance.json",
            "  permutation_importance_scoring: average_precision",
            "  permutation_importance_repeats: 3",
        ]
    )
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


class FakeSplitConfig:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state


class FakeXGBoostModelConfig:
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


class FakeXGBoostCreditRiskModel:
    def __init__(
        self,
        *,
        id_column: str,
        target_column: str,
        model_version: str,
        schema_version: str,
        config: FakeXGBoostModelConfig,
        native_importance_types: tuple[str, ...],
    ) -> None:
        self.id_column = id_column
        self.target_column = target_column
        self.model_version = model_version
        self.schema_version = schema_version
        self.config = config
        self.native_importance_types = native_importance_types
        self.feature_schema = FakeFeatureSchema(schema_version)
        self.training_summary_ = None
        self.evaluation_history_ = None

    def fit(self, dataframe: pd.DataFrame, *, eval_dataframe: pd.DataFrame | None = None, verbose: bool = False):
        self.training_summary_ = {
            "row_count": int(len(dataframe)),
            "positive_rate": float((dataframe[self.target_column] == 1).mean()),
            "eval_row_count": int(len(eval_dataframe)) if eval_dataframe is not None else 0,
            "best_iteration": 12,
            "best_score": 0.271,
        }
        self.evaluation_history_ = {
            "validation_0": {
                "aucpr": [0.20, 0.25, 0.27],
                "auc": [0.70, 0.74, 0.78],
            }
        }
        self.verbose_ = verbose
        return self

    def predict_proba(self, dataframe: pd.DataFrame) -> pd.Series:
        values = [0.8 if value == 1 else 0.2 for value in dataframe[self.target_column].tolist()]
        return pd.Series(values, index=dataframe.index, name="predicted_probability")

    def export_native_importance(self) -> dict[str, dict[str, float]]:
        return {
            "gain": {"DebtRatio": 3.0, "MonthlyIncome": 2.0},
            "weight": {"DebtRatio": 30.0, "MonthlyIncome": 20.0},
            "cover": {"DebtRatio": 300.0, "MonthlyIncome": 200.0},
        }

    def build_artifact_metadata(self) -> dict[str, object]:
        return {
            "model_name": "xgboost_credit_risk",
            "model_version": self.model_version,
            "schema_version": self.schema_version,
            "native_importance_types": list(self.native_importance_types),
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake xgboost artifact", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path):
        del path
        return cls(
            id_column="Unnamed: 0",
            target_column="SeriousDlqin2yrs",
            model_version="0.2.0",
            schema_version="1.1.0",
            config=FakeXGBoostModelConfig(),
            native_importance_types=("gain", "weight", "cover"),
        )


def _fake_split_training_data(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    config: FakeSplitConfig,
):
    del target_column, config
    return SimpleNamespace(
        train=dataframe.iloc[:6].reset_index(drop=True),
        validation=dataframe.iloc[6:9].reset_index(drop=True),
        test=dataframe.iloc[9:12].reset_index(drop=True),
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
        "average_precision": 0.62,
        "roc_auc": 0.79,
        "ks_statistic": 0.48,
        "brier_score": 0.17,
        "accuracy": 0.75,
        "precision": 0.67,
        "recall": 0.80,
        "f1": 0.73,
        "confusion_matrix": {
            "counts": {"matrix": [[1, 0], [0, 1]]},
        },
        "precision_recall_curve": {
            "point_count": 3,
            "threshold_count": 2,
            "baseline_positive_rate": float((pd.Series(y_true) == 1).mean()),
            "precision": [0.3333333333333333, 0.5, 1.0],
            "recall": [1.0, 1.0, 0.0],
            "thresholds": [0.2, 0.8],
        },
        "roc_curve": {
            "point_count": 3,
            "threshold_count": 3,
            "false_positive_rate": [0.0, 0.0, 1.0],
            "true_positive_rate": [0.0, 1.0, 1.0],
            "thresholds": [float("inf"), 0.8, 0.2],
        },
        "mean_score": float(pd.Series(y_score).mean()),
    }


def _fake_evaluate_binary_classifier_from_scores(y_true, y_score, *, threshold: float):
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    y_score_series = pd.Series(y_score).reset_index(drop=True)
    mean_score = float(y_score_series.mean())
    positive_mean = float(y_score_series[y_true_series == 1].mean())
    negative_mean = float(y_score_series[y_true_series == 0].mean())
    separation = positive_mean - negative_mean
    return {
        "row_count": len(y_true),
        "positive_rate": float((y_true_series == 1).mean()),
        "threshold": threshold,
        "average_precision": separation,
        "roc_auc": separation + 0.1,
        "ks_statistic": separation + 0.2,
        "brier_score": 0.17,
        "accuracy": 0.75,
        "precision": 0.67,
        "recall": 0.80,
        "f1": 0.73,
        "confusion_matrix": {
            "counts": {"matrix": [[1, 0], [0, 1]]},
        },
        "precision_recall_curve": {
            "point_count": 3,
            "threshold_count": 2,
            "baseline_positive_rate": float((y_true_series == 1).mean()),
            "precision": [0.3333333333333333, 0.5, 1.0],
            "recall": [1.0, 1.0, 0.0],
            "thresholds": [0.3, 0.9],
        },
        "roc_curve": {
            "point_count": 3,
            "threshold_count": 3,
            "false_positive_rate": [0.0, 0.0, 1.0],
            "true_positive_rate": [0.0, 1.0, 1.0],
            "thresholds": [float("inf"), 0.9, 0.3],
        },
        "mean_score": mean_score,
    }


def test_train_xgboost_main_writes_expected_artifacts(monkeypatch, tmp_path: Path) -> None:
    module = _load_train_xgboost_module()
    output_dir = tmp_path / "custom_output"
    config_path = tmp_path / "model_xgb.yaml"
    training_data = _build_training_dataframe()
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
    monkeypatch.setattr(module, "build_repository", lambda settings: FakeRepository(training_data))
    monkeypatch.setattr(
        module,
        "_load_training_modules",
        lambda: (
            _fake_evaluate_binary_classifier,
            FakeSplitConfig,
            _fake_build_split_metadata,
            _fake_split_training_data,
            (FakeXGBoostCreditRiskModel, FakeXGBoostModelConfig),
        ),
    )

    result = module.main()

    assert result == 0
    assert (output_dir / "xgboost_credit_risk.joblib").exists()
    assert (output_dir / "feature_schema.json").exists()
    assert (output_dir / "split_metadata.json").exists()
    assert (output_dir / "xgboost_metrics.json").exists()
    assert (output_dir / "run_summary.json").exists()
    assert (output_dir / "learning_curve.json").exists()
    assert (output_dir / "curves.json").exists()
    assert (output_dir / "native_feature_importance.json").exists()
    assert (output_dir / "permutation_importance.json").exists()
    assert (output_dir / "tuning_results.json").exists()
    assert (output_dir / "xgboost_config_snapshot.json").exists()

    feature_schema_payload = json.loads((output_dir / "feature_schema.json").read_text())
    split_metadata_payload = json.loads((output_dir / "split_metadata.json").read_text())
    metrics_payload = json.loads((output_dir / "xgboost_metrics.json").read_text())
    run_summary_payload = json.loads((output_dir / "run_summary.json").read_text())
    tuning_results_payload = json.loads((output_dir / "tuning_results.json").read_text())
    learning_curve_payload = json.loads((output_dir / "learning_curve.json").read_text())
    curves_payload = json.loads((output_dir / "curves.json").read_text())
    permutation_importance_payload = json.loads(
        (output_dir / "permutation_importance.json").read_text()
    )

    assert feature_schema_payload["feature_schema"]["schema_version"] == "1.1.0"
    assert split_metadata_payload["dataset_fingerprint"] == "fixture-fingerprint"
    assert split_metadata_payload["partitions"]["train"]["row_count"] == 6
    assert metrics_payload["threshold"] == 0.4
    assert metrics_payload["partitions"]["validation"]["average_precision"] == 0.62
    assert metrics_payload["model_version"] == "0.2.0"
    assert run_summary_payload["artifact_metadata"]["schema_version"] == "1.1.0"
    assert run_summary_payload["training_summary"]["row_count"] == 6
    assert run_summary_payload["reload_check"]["status"] == "passed"
    assert run_summary_payload["reload_check"]["row_count"] == 3
    assert tuning_results_payload["status"] == "not_run"
    assert learning_curve_payload["evaluation_history"]["validation_0"]["aucpr"][-1] == 0.27
    assert curves_payload["partitions"]["validation"]["precision_recall_curve"]["baseline_positive_rate"] == 0.3333333333333333
    assert curves_payload["partitions"]["validation"]["roc_curve"]["false_positive_rate"][0] == 0.0
    assert permutation_importance_payload["partition"] == "validation"
    assert permutation_importance_payload["scoring_metric"] == "average_precision"
    assert permutation_importance_payload["n_repeats"] == 3
    assert len(permutation_importance_payload["features"]) == 2
    assert permutation_importance_payload["features"][0]["feature_name"] in {"MonthlyIncome", "DebtRatio"}


def test_train_xgboost_main_selects_tuned_candidate_when_search_improves_metric(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_train_xgboost_module()
    output_dir = tmp_path / "custom_output"
    config_path = tmp_path / "model_xgb.yaml"
    training_data = _build_training_dataframe()
    _write_config(config_path, include_tuning=True)

    class TuningAwareFakeXGBoostCreditRiskModel(FakeXGBoostCreditRiskModel):
        def predict_proba(self, dataframe: pd.DataFrame) -> pd.Series:
            max_depth = self.config.kwargs.get("max_depth", 4)
            if max_depth >= 5:
                positive_score = 0.90
                negative_score = 0.30
            elif max_depth <= 3:
                positive_score = 0.70
                negative_score = 0.40
            else:
                positive_score = 0.75
                negative_score = 0.35
            values = [
                positive_score if value == 1 else negative_score
                for value in dataframe[self.target_column].tolist()
            ]
            return pd.Series(values, index=dataframe.index, name="predicted_probability")

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
    monkeypatch.setattr(module, "build_repository", lambda settings: FakeRepository(training_data))
    monkeypatch.setattr(
        module,
        "_load_training_modules",
        lambda: (
            _fake_evaluate_binary_classifier_from_scores,
            FakeSplitConfig,
            _fake_build_split_metadata,
            _fake_split_training_data,
            (TuningAwareFakeXGBoostCreditRiskModel, FakeXGBoostModelConfig),
        ),
    )

    result = module.main()

    assert result == 0
    metrics_payload = json.loads((output_dir / "xgboost_metrics.json").read_text())
    run_summary_payload = json.loads((output_dir / "run_summary.json").read_text())
    tuning_results_payload = json.loads((output_dir / "tuning_results.json").read_text())

    assert metrics_payload["selected_candidate_source"] == "tuned_search"
    assert run_summary_payload["selected_candidate_source"] == "tuned_search"
    assert tuning_results_payload["status"] == "completed"
    assert tuning_results_payload["selected_candidate_source"] == "tuned_search"
    assert tuning_results_payload["search"]["successful_candidates"] == 2
    assert tuning_results_payload["selected_candidate"]["params"]["max_depth"] == 5
    assert tuning_results_payload["search"]["leaderboard"][0]["params"]["max_depth"] == 5


def test_train_xgboost_main_selects_scale_pos_weight_variant_when_it_improves_metric(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_train_xgboost_module()
    output_dir = tmp_path / "custom_output"
    config_path = tmp_path / "model_xgb.yaml"
    training_data = _build_training_dataframe()
    _write_config(config_path, include_scale_pos_weight=True)

    class ScaleAwareFakeXGBoostCreditRiskModel(FakeXGBoostCreditRiskModel):
        def predict_proba(self, dataframe: pd.DataFrame) -> pd.Series:
            scale_pos_weight = float(self.config.kwargs.get("scale_pos_weight", 1.0))
            if scale_pos_weight > 1.0:
                positive_score = 0.92
                negative_score = 0.20
            else:
                positive_score = 0.70
                negative_score = 0.35
            values = [
                positive_score if value == 1 else negative_score
                for value in dataframe[self.target_column].tolist()
            ]
            return pd.Series(values, index=dataframe.index, name="predicted_probability")

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
    monkeypatch.setattr(module, "build_repository", lambda settings: FakeRepository(training_data))
    monkeypatch.setattr(
        module,
        "_load_training_modules",
        lambda: (
            _fake_evaluate_binary_classifier_from_scores,
            FakeSplitConfig,
            _fake_build_split_metadata,
            _fake_split_training_data,
            (ScaleAwareFakeXGBoostCreditRiskModel, FakeXGBoostModelConfig),
        ),
    )

    result = module.main()

    assert result == 0
    metrics_payload = json.loads((output_dir / "xgboost_metrics.json").read_text())
    run_summary_payload = json.loads((output_dir / "run_summary.json").read_text())
    tuning_results_payload = json.loads((output_dir / "tuning_results.json").read_text())

    assert metrics_payload["selected_candidate_source"] == "scale_pos_weight_variant"
    assert run_summary_payload["selected_candidate_source"] == "scale_pos_weight_variant"
    assert tuning_results_payload["class_imbalance_experiments"]["status"] == "completed"
    assert (
        tuning_results_payload["class_imbalance_experiments"]["selected_candidate_source"]
        == "scale_pos_weight_variant"
    )
    assert (
        tuning_results_payload["class_imbalance_experiments"]["scale_pos_weight_variant"][
            "computed_scale_pos_weight"
        ]
        == 2.0
    )
    assert (
        tuning_results_payload["class_imbalance_experiments"]["scale_pos_weight_variant"][
            "train_distribution"
        ]["positive_count"]
        == 2
    )


def test_train_xgboost_main_returns_error_when_repository_validation_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_train_xgboost_module()
    output_dir = tmp_path / "custom_output"
    config_path = tmp_path / "model_xgb.yaml"
    training_data = _build_training_dataframe()
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
            (FakeXGBoostCreditRiskModel, FakeXGBoostModelConfig),
        ),
    )

    result = module.main()

    assert result == 1
    assert not (output_dir / "xgboost_metrics.json").exists()


def test_train_xgboost_main_returns_error_when_model_fit_raises_import_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_train_xgboost_module()
    output_dir = tmp_path / "custom_output"
    config_path = tmp_path / "model_xgb.yaml"
    training_data = _build_training_dataframe()
    _write_config(config_path)

    class FailingXGBoostCreditRiskModel(FakeXGBoostCreditRiskModel):
        def fit(self, dataframe: pd.DataFrame, *, eval_dataframe: pd.DataFrame | None = None, verbose: bool = False):
            del dataframe, eval_dataframe, verbose
            raise ImportError("libomp runtime is missing.")

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
    monkeypatch.setattr(module, "build_repository", lambda settings: FakeRepository(training_data))
    monkeypatch.setattr(
        module,
        "_load_training_modules",
        lambda: (
            _fake_evaluate_binary_classifier,
            FakeSplitConfig,
            _fake_build_split_metadata,
            _fake_split_training_data,
            (FailingXGBoostCreditRiskModel, FakeXGBoostModelConfig),
        ),
    )

    result = module.main()

    assert result == 1
    assert not (output_dir / "run_summary.json").exists()


def test_train_xgboost_main_returns_error_when_repository_load_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_train_xgboost_module()
    output_dir = tmp_path / "custom_output"
    config_path = tmp_path / "model_xgb.yaml"
    _write_config(config_path)

    class FailingRepository(FakeRepository):
        def load_training_data(self) -> pd.DataFrame:
            raise RepositoryValidationError("Training contract failed.")

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
            (FakeXGBoostCreditRiskModel, FakeXGBoostModelConfig),
        ),
    )

    result = module.main()

    assert result == 1
    assert not (output_dir / "xgboost_config_snapshot.json").exists()
