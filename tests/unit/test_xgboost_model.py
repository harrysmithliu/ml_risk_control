from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ml_risk_control.models.xgboost_model as xgb_module
from ml_risk_control.models.xgboost_model import (
    XGBoostCreditRiskModel,
    XGBoostModelConfig,
)


def _build_training_dataframe(row_count: int = 40) -> pd.DataFrame:
    rows: list[dict[str, float | int | None]] = []
    for index in range(row_count):
        rows.append(
            {
                "Unnamed: 0": index + 1,
                "SeriousDlqin2yrs": 1 if index % 5 in {0, 1} else 0,
                "RevolvingUtilizationOfUnsecuredLines": 1.30 if index == 0 else 0.09 + index * 0.01,
                "age": 16 if index == 0 else 29 + (index % 40),
                "NumberOfTime30-59DaysPastDueNotWorse": 22 if index == 0 else index % 4,
                "DebtRatio": 6.8 if index == 0 else 0.12 + index * 0.02,
                "MonthlyIncome": None if index in {0, 8, 16} else 3200 + index * 90,
                "NumberOfOpenCreditLinesAndLoans": 3 + (index % 7),
                "NumberOfTimes90DaysLate": 21 if index == 0 else index % 3,
                "NumberRealEstateLoansOrLines": index % 4,
                "NumberOfTime60-89DaysPastDueNotWorse": 23 if index == 0 else index % 2,
                "NumberOfDependents": None if index in {0, 10} else index % 3,
            }
        )
    return pd.DataFrame(rows)


class FakeBooster:
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names

    def get_score(self, importance_type: str = "gain") -> dict[str, float]:
        multiplier = {
            "gain": 1.0,
            "weight": 10.0,
            "cover": 100.0,
        }[importance_type]
        return {
            feature_name: float((len(self.feature_names) - index) * multiplier)
            for index, feature_name in enumerate(self.feature_names)
        }


class FakeXGBClassifier:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.best_iteration = 17
        self.best_score = 0.314159
        self._feature_names: list[str] = []
        self._evals_result = {
            "validation_0": {
                "aucpr": [0.21, 0.29, 0.31],
                "auc": [0.70, 0.75, 0.79],
                "logloss": [0.60, 0.55, 0.51],
            }
        }

    def fit(self, X, y, eval_set=None, verbose=False):
        self._feature_names = list(X.columns)
        self.fit_row_count_ = int(len(X))
        self.fit_positive_rate_ = float((pd.Series(y) == 1).mean())
        self.received_eval_set_ = eval_set
        self.received_verbose_ = verbose
        return self

    def predict_proba(self, X):
        raw_scores = pd.DataFrame(X).sum(axis=1).to_numpy(dtype=float)
        probabilities = 1.0 / (1.0 + np.exp(-raw_scores / 10.0))
        probabilities = np.clip(probabilities, 0.01, 0.99)
        return np.column_stack([1.0 - probabilities, probabilities])

    def evals_result(self) -> dict[str, dict[str, list[float]]]:
        return self._evals_result

    def get_booster(self) -> FakeBooster:
        return FakeBooster(self._feature_names)


def test_xgboost_model_config_to_model_kwargs_returns_expected_defaults() -> None:
    config = XGBoostModelConfig()

    params = config.to_model_kwargs()

    assert params["objective"] == "binary:logistic"
    assert params["tree_method"] == "hist"
    assert params["device"] == "cpu"
    assert params["early_stopping_rounds"] == 50


def test_xgboost_model_fit_populates_training_summary_and_eval_history(monkeypatch) -> None:
    monkeypatch.setattr(xgb_module, "_load_xgb_classifier", lambda: FakeXGBClassifier)
    dataframe = _build_training_dataframe()
    train_frame = dataframe.iloc[:30].reset_index(drop=True)
    eval_frame = dataframe.iloc[30:].reset_index(drop=True)
    model = XGBoostCreditRiskModel(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
    )

    result = model.fit(train_frame, eval_dataframe=eval_frame, verbose=True)

    assert result is model
    assert model.preprocessing_pipeline_ is not None
    assert model.classifier_ is not None
    assert model.training_summary_ is not None
    assert model.training_summary_["row_count"] == 30
    assert model.training_summary_["eval_row_count"] == 10
    assert model.training_summary_["classifier_class"] == "FakeXGBClassifier"
    assert model.training_summary_["best_iteration"] == 17
    assert model.training_summary_["best_score"] == pytest.approx(0.314159)
    assert model.evaluation_history_ == model.classifier_.evals_result()
    assert model.classifier_.received_eval_set_ is not None
    assert len(model.classifier_.received_eval_set_) == 1
    assert model.classifier_.received_verbose_ is True
    assert model.transformed_feature_names_ == model.feature_schema.model_input_features


def test_xgboost_model_predict_methods_return_named_outputs(monkeypatch) -> None:
    monkeypatch.setattr(xgb_module, "_load_xgb_classifier", lambda: FakeXGBClassifier)
    dataframe = _build_training_dataframe()
    model = XGBoostCreditRiskModel(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
    ).fit(dataframe.iloc[:30].reset_index(drop=True), eval_dataframe=dataframe.iloc[30:].reset_index(drop=True))

    probabilities = model.predict_proba(dataframe)
    predictions = model.predict(dataframe, threshold=0.45)

    assert probabilities.name == "predicted_probability"
    assert predictions.name == "predicted_label"
    assert len(probabilities) == len(dataframe)
    assert probabilities.between(0.0, 1.0).all()
    assert set(predictions.unique()).issubset({0, 1})


def test_xgboost_model_score_records_can_include_identifier(monkeypatch) -> None:
    monkeypatch.setattr(xgb_module, "_load_xgb_classifier", lambda: FakeXGBClassifier)
    dataframe = _build_training_dataframe()
    model = XGBoostCreditRiskModel(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
    ).fit(dataframe.iloc[:30].reset_index(drop=True), eval_dataframe=dataframe.iloc[30:].reset_index(drop=True))

    scored = model.score_records(dataframe, threshold=0.5, include_identifier=True)

    assert scored.columns.tolist() == [
        "Unnamed: 0",
        "predicted_probability",
        "predicted_label",
    ]
    assert scored["Unnamed: 0"].tolist() == dataframe["Unnamed: 0"].tolist()


def test_xgboost_model_exports_native_importance_and_metadata(monkeypatch) -> None:
    monkeypatch.setattr(xgb_module, "_load_xgb_classifier", lambda: FakeXGBClassifier)
    dataframe = _build_training_dataframe()
    model = XGBoostCreditRiskModel(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
        model_version="0.2.0",
        schema_version="1.1.0",
    ).fit(dataframe.iloc[:30].reset_index(drop=True), eval_dataframe=dataframe.iloc[30:].reset_index(drop=True))

    native_importance = model.export_native_importance()
    metadata = model.build_artifact_metadata()

    assert sorted(native_importance.keys()) == ["cover", "gain", "weight"]
    assert set(native_importance["gain"].keys()) == set(model.feature_schema.model_input_features)
    assert "MonthlyIncome_missing" in native_importance["cover"]
    assert metadata["model_name"] == "xgboost_credit_risk"
    assert metadata["model_version"] == "0.2.0"
    assert metadata["schema_version"] == "1.1.0"
    assert metadata["training_summary"]["best_iteration"] == 17
    assert metadata["evaluation_history"]["validation_0"]["aucpr"][-1] == pytest.approx(0.31)
    assert metadata["transformed_feature_names"] == list(model.feature_schema.model_input_features)


def test_xgboost_model_save_and_load_round_trip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(xgb_module, "_load_xgb_classifier", lambda: FakeXGBClassifier)
    dataframe = _build_training_dataframe()
    model = XGBoostCreditRiskModel(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
        config=XGBoostModelConfig(n_estimators=300, max_depth=5),
    ).fit(dataframe.iloc[:30].reset_index(drop=True), eval_dataframe=dataframe.iloc[30:].reset_index(drop=True))
    artifact_path = tmp_path / "artifacts" / "xgboost.joblib"

    saved_path = model.save(artifact_path)
    loaded = XGBoostCreditRiskModel.load(saved_path)

    original_probabilities = model.predict_proba(dataframe).tolist()
    loaded_probabilities = loaded.predict_proba(dataframe).tolist()

    assert saved_path.exists()
    assert loaded.model_version == model.model_version
    assert loaded.schema_version == model.schema_version
    assert loaded.feature_schema.to_dict() == model.feature_schema.to_dict()
    assert loaded.training_summary_ == model.training_summary_
    assert loaded.evaluation_history_ == model.evaluation_history_
    assert loaded.transformed_feature_names_ == model.transformed_feature_names_
    assert loaded_probabilities == pytest.approx(original_probabilities)


def test_xgboost_model_requires_fit_before_inference() -> None:
    dataframe = _build_training_dataframe()
    model = XGBoostCreditRiskModel(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
    )

    with pytest.raises(ValueError, match="must be fitted or loaded before inference"):
        model.predict_proba(dataframe)


def test_xgboost_model_raises_when_training_target_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(xgb_module, "_load_xgb_classifier", lambda: FakeXGBClassifier)
    dataframe = _build_training_dataframe().drop(columns=["SeriousDlqin2yrs"])
    model = XGBoostCreditRiskModel(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
    )

    with pytest.raises(ValueError, match="Target column 'SeriousDlqin2yrs' is missing"):
        model.fit(dataframe)
