from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_risk_control.models.baseline import (
    LogisticRegressionBaseline,
    LogisticRegressionBaselineConfig,
)


def _build_training_dataframe(row_count: int = 60) -> pd.DataFrame:
    rows: list[dict[str, float | int | None]] = []
    for index in range(row_count):
        rows.append(
            {
                "Unnamed: 0": index + 1,
                "SeriousDlqin2yrs": 1 if index % 6 in {0, 1} else 0,
                "RevolvingUtilizationOfUnsecuredLines": 1.40 if index == 0 else 0.08 + index * 0.01,
                "age": 15 if index == 0 else 28 + (index % 35),
                "NumberOfTime30-59DaysPastDueNotWorse": 26 if index == 0 else index % 4,
                "DebtRatio": 7.2 if index == 0 else 0.15 + index * 0.02,
                "MonthlyIncome": None if index in {0, 9, 18} else 2800 + index * 110,
                "NumberOfOpenCreditLinesAndLoans": 2 + (index % 8),
                "NumberOfTimes90DaysLate": 21 if index == 0 else index % 3,
                "NumberRealEstateLoansOrLines": index % 4,
                "NumberOfTime60-89DaysPastDueNotWorse": 23 if index == 0 else index % 2,
                "NumberOfDependents": None if index in {0, 11} else index % 3,
            }
        )
    return pd.DataFrame(rows)


def test_logistic_regression_baseline_fit_populates_pipeline_and_training_summary() -> None:
    dataframe = _build_training_dataframe()
    baseline = LogisticRegressionBaseline(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
    )

    result = baseline.fit(dataframe)

    assert result is baseline
    assert baseline.pipeline_ is not None
    assert list(baseline.pipeline_.named_steps.keys()) == ["preprocessing", "classifier"]
    assert baseline.training_summary_ is not None
    assert baseline.training_summary_["row_count"] == 60
    assert baseline.training_summary_["positive_rate"] == pytest.approx(20 / 60)
    assert baseline.training_summary_["classifier_class"] == "LogisticRegression"


def test_logistic_regression_baseline_predict_methods_return_named_outputs() -> None:
    dataframe = _build_training_dataframe()
    baseline = LogisticRegressionBaseline(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
    ).fit(dataframe)

    probabilities = baseline.predict_proba(dataframe)
    predictions = baseline.predict(dataframe, threshold=0.4)

    assert probabilities.name == "predicted_probability"
    assert predictions.name == "predicted_label"
    assert len(probabilities) == len(dataframe)
    assert len(predictions) == len(dataframe)
    assert probabilities.between(0.0, 1.0).all()
    assert set(predictions.unique()).issubset({0, 1})


def test_logistic_regression_baseline_score_records_can_include_or_exclude_identifier() -> None:
    dataframe = _build_training_dataframe()
    baseline = LogisticRegressionBaseline(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
    ).fit(dataframe)

    scored_with_id = baseline.score_records(dataframe, threshold=0.45, include_identifier=True)
    scored_without_id = baseline.score_records(
        dataframe,
        threshold=0.45,
        include_identifier=False,
    )

    assert scored_with_id.columns.tolist() == [
        "Unnamed: 0",
        "predicted_probability",
        "predicted_label",
    ]
    assert scored_without_id.columns.tolist() == [
        "predicted_probability",
        "predicted_label",
    ]
    assert scored_with_id["Unnamed: 0"].tolist() == dataframe["Unnamed: 0"].tolist()


def test_logistic_regression_baseline_build_artifact_metadata_reflects_training_state() -> None:
    dataframe = _build_training_dataframe()
    baseline = LogisticRegressionBaseline(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
        model_version="0.1.0",
        schema_version="1.0.0",
        config=LogisticRegressionBaselineConfig(max_iter=500, class_weight="balanced"),
    ).fit(dataframe)

    metadata = baseline.build_artifact_metadata()

    assert metadata["model_name"] == "logistic_regression_baseline"
    assert metadata["model_version"] == "0.1.0"
    assert metadata["schema_version"] == "1.0.0"
    assert metadata["pipeline_steps"] == ["preprocessing", "classifier"]
    assert metadata["classifier_config"]["max_iter"] == 500
    assert metadata["classifier_config"]["class_weight"] == "balanced"
    assert "MonthlyIncome_missing" in metadata["model_input_features"]
    assert metadata["training_summary"]["row_count"] == 60


def test_logistic_regression_baseline_save_and_load_round_trip(tmp_path: Path) -> None:
    dataframe = _build_training_dataframe()
    baseline = LogisticRegressionBaseline(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
        config=LogisticRegressionBaselineConfig(max_iter=300),
    ).fit(dataframe)
    artifact_path = tmp_path / "artifacts" / "baseline.joblib"

    saved_path = baseline.save(artifact_path)
    loaded = LogisticRegressionBaseline.load(saved_path)

    original_probabilities = baseline.predict_proba(dataframe).tolist()
    loaded_probabilities = loaded.predict_proba(dataframe).tolist()

    assert saved_path.exists()
    assert loaded.model_version == baseline.model_version
    assert loaded.schema_version == baseline.schema_version
    assert loaded.feature_schema.to_dict() == baseline.feature_schema.to_dict()
    assert loaded.training_summary_ == baseline.training_summary_
    assert loaded_probabilities == pytest.approx(original_probabilities)


def test_logistic_regression_baseline_requires_fit_before_inference() -> None:
    dataframe = _build_training_dataframe()
    baseline = LogisticRegressionBaseline(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
    )

    with pytest.raises(ValueError, match="must be fitted or loaded before inference"):
        baseline.predict_proba(dataframe)


def test_logistic_regression_baseline_raises_when_training_target_is_missing() -> None:
    dataframe = _build_training_dataframe().drop(columns=["SeriousDlqin2yrs"])
    baseline = LogisticRegressionBaseline(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
    )

    with pytest.raises(ValueError, match="Target column 'SeriousDlqin2yrs' is missing"):
        baseline.fit(dataframe)
