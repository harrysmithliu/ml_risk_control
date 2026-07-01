from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_risk_control.models.torch_model import (
    TorchMLPConfig,
    TorchMLPCreditRiskModel,
)


def _build_training_dataframe(row_count: int = 48) -> pd.DataFrame:
    rows: list[dict[str, float | int | None]] = []
    for index in range(row_count):
        is_positive = index % 4 == 0
        rows.append(
            {
                "Unnamed: 0": index + 1,
                "SeriousDlqin2yrs": 1 if is_positive else 0,
                "RevolvingUtilizationOfUnsecuredLines": 0.78 if is_positive else 0.12 + index * 0.003,
                "age": 32 if is_positive else 45 + (index % 12),
                "NumberOfTime30-59DaysPastDueNotWorse": 3 if is_positive else index % 2,
                "DebtRatio": 1.10 if is_positive else 0.18 + index * 0.01,
                "MonthlyIncome": None if index in {0, 8, 16} else (2400 if is_positive else 5200 + index * 45),
                "NumberOfOpenCreditLinesAndLoans": 3 if is_positive else 8 + (index % 4),
                "NumberOfTimes90DaysLate": 2 if is_positive else 0,
                "NumberRealEstateLoansOrLines": 0 if is_positive else 1 + (index % 3),
                "NumberOfTime60-89DaysPastDueNotWorse": 1 if is_positive else 0,
                "NumberOfDependents": None if index in {0, 10} else (2 if is_positive else index % 2),
            }
        )
    return pd.DataFrame(rows)


def _build_config() -> TorchMLPConfig:
    return TorchMLPConfig(
        hidden_dims=(16, 8),
        dropout=0.05,
        learning_rate=0.01,
        batch_size=8,
        max_epochs=6,
        patience=3,
        min_delta=1e-4,
        weight_decay=0.0,
        positive_class_weight_strategy="auto_from_train_ratio",
        random_state=42,
    )


def test_torch_model_config_to_dict_returns_expected_values() -> None:
    config = _build_config()

    payload = config.to_dict()

    assert payload["hidden_dims"] == (16, 8)
    assert payload["batch_size"] == 8
    assert payload["positive_class_weight_strategy"] == "auto_from_train_ratio"


def test_torch_model_fit_populates_training_summary_and_history() -> None:
    dataframe = _build_training_dataframe()
    train_frame = dataframe.iloc[:36].reset_index(drop=True)
    eval_frame = dataframe.iloc[36:].reset_index(drop=True)
    model = TorchMLPCreditRiskModel(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
        config=_build_config(),
    )

    result = model.fit(train_frame, eval_dataframe=eval_frame, verbose=False)

    assert result is model
    assert model.preprocessing_pipeline_ is not None
    assert model.classifier_ is not None
    assert model.training_summary_ is not None
    assert model.training_history_ is not None
    assert model.training_summary_["row_count"] == 36
    assert model.training_summary_["eval_row_count"] == 12
    assert model.training_summary_["best_epoch"] >= 1
    assert model.training_summary_["effective_positive_class_weight"] > 0.0
    assert model.training_history_["epochs_completed"] >= 1
    assert len(model.training_history_["history"]) == model.training_history_["epochs_completed"]
    assert model.transformed_feature_names_ == model.feature_schema.model_input_features


def test_torch_model_predict_methods_return_named_outputs() -> None:
    dataframe = _build_training_dataframe()
    model = TorchMLPCreditRiskModel(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
        config=_build_config(),
    ).fit(
        dataframe.iloc[:36].reset_index(drop=True),
        eval_dataframe=dataframe.iloc[36:].reset_index(drop=True),
    )

    probabilities = model.predict_proba(dataframe)
    predictions = model.predict(dataframe, threshold=0.45)

    assert probabilities.name == "predicted_probability"
    assert predictions.name == "predicted_label"
    assert len(probabilities) == len(dataframe)
    assert probabilities.between(0.0, 1.0).all()
    assert set(predictions.unique()).issubset({0, 1})


def test_torch_model_score_records_can_include_identifier() -> None:
    dataframe = _build_training_dataframe()
    model = TorchMLPCreditRiskModel(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
        config=_build_config(),
    ).fit(
        dataframe.iloc[:36].reset_index(drop=True),
        eval_dataframe=dataframe.iloc[36:].reset_index(drop=True),
    )

    scored = model.score_records(dataframe, threshold=0.5, include_identifier=True)

    assert scored.columns.tolist() == [
        "Unnamed: 0",
        "predicted_probability",
        "predicted_label",
    ]
    assert scored["Unnamed: 0"].tolist() == dataframe["Unnamed: 0"].tolist()


def test_torch_model_builds_artifact_metadata() -> None:
    dataframe = _build_training_dataframe()
    model = TorchMLPCreditRiskModel(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
        model_version="0.2.0",
        schema_version="1.1.0",
        config=_build_config(),
    ).fit(
        dataframe.iloc[:36].reset_index(drop=True),
        eval_dataframe=dataframe.iloc[36:].reset_index(drop=True),
    )

    metadata = model.build_artifact_metadata()

    assert metadata["model_name"] == "torch_mlp_challenger"
    assert metadata["model_version"] == "0.2.0"
    assert metadata["schema_version"] == "1.1.0"
    assert metadata["training_summary"]["best_epoch"] >= 1
    assert metadata["input_dim"] == len(model.feature_schema.model_input_features)
    assert metadata["transformed_feature_names"] == list(model.feature_schema.model_input_features)


def test_torch_model_save_and_load_round_trip(tmp_path: Path) -> None:
    dataframe = _build_training_dataframe()
    model = TorchMLPCreditRiskModel(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
        config=_build_config(),
    ).fit(
        dataframe.iloc[:36].reset_index(drop=True),
        eval_dataframe=dataframe.iloc[36:].reset_index(drop=True),
    )
    artifact_path = tmp_path / "artifacts" / "torch_mlp_challenger.pt"

    saved_path = model.save(artifact_path)
    loaded = TorchMLPCreditRiskModel.load(saved_path)

    original_probabilities = model.predict_proba(dataframe).tolist()
    loaded_probabilities = loaded.predict_proba(dataframe).tolist()

    assert saved_path.exists()
    assert loaded.model_version == model.model_version
    assert loaded.schema_version == model.schema_version
    assert loaded.feature_schema.to_dict() == model.feature_schema.to_dict()
    assert loaded.training_summary_ == model.training_summary_
    assert loaded.training_history_ == model.training_history_
    assert loaded.transformed_feature_names_ == model.transformed_feature_names_
    assert loaded_probabilities == pytest.approx(original_probabilities, rel=1e-5, abs=1e-5)


def test_torch_model_requires_fit_before_inference() -> None:
    dataframe = _build_training_dataframe()
    model = TorchMLPCreditRiskModel(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
        config=_build_config(),
    )

    with pytest.raises(ValueError, match="must be fitted or loaded before inference"):
        model.predict_proba(dataframe)


def test_torch_model_raises_when_training_target_is_missing() -> None:
    dataframe = _build_training_dataframe().drop(columns=["SeriousDlqin2yrs"])
    model = TorchMLPCreditRiskModel(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
        config=_build_config(),
    )

    with pytest.raises(ValueError, match="Target column 'SeriousDlqin2yrs' is missing"):
        model.fit(dataframe)
