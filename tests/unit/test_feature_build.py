from __future__ import annotations

import pandas as pd
import pytest

from ml_risk_control.features.build import (
    CreditRiskFeatureBuilder,
    SplitConfig,
    build_feature_schema,
    build_split_metadata,
    create_preprocessing_pipeline,
    get_model_feature_columns,
    split_training_data,
)


def _build_training_dataframe(row_count: int = 40) -> pd.DataFrame:
    rows: list[dict[str, float | int | None]] = []
    for index in range(row_count):
        rows.append(
            {
                "Unnamed: 0": index + 1,
                "SeriousDlqin2yrs": 1 if index % 5 == 0 else 0,
                "RevolvingUtilizationOfUnsecuredLines": 1.25 if index == 0 else 0.10 + index * 0.01,
                "age": 12 if index == 0 else 30 + index,
                "NumberOfTime30-59DaysPastDueNotWorse": 25 if index == 0 else index % 4,
                "DebtRatio": 6.5 if index == 0 else 0.20 + index * 0.03,
                "MonthlyIncome": None if index in {0, 7} else 3000 + index * 100,
                "NumberOfOpenCreditLinesAndLoans": 3 + (index % 7),
                "NumberOfTimes90DaysLate": 22 if index == 0 else index % 3,
                "NumberRealEstateLoansOrLines": index % 4,
                "NumberOfTime60-89DaysPastDueNotWorse": 24 if index == 0 else index % 2,
                "NumberOfDependents": None if index in {0, 9} else index % 3,
            }
        )
    return pd.DataFrame(rows)


def test_get_model_feature_columns_excludes_identifier_and_target() -> None:
    feature_columns = get_model_feature_columns(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
        feature_columns=(
            "Unnamed: 0",
            "SeriousDlqin2yrs",
            "MonthlyIncome",
            "DebtRatio",
        ),
    )

    assert feature_columns == ("MonthlyIncome", "DebtRatio")


def test_split_training_data_is_reproducible_and_preserves_all_rows() -> None:
    dataframe = _build_training_dataframe()
    config = SplitConfig(random_state=42)

    first_split = split_training_data(
        dataframe,
        target_column="SeriousDlqin2yrs",
        config=config,
    )
    second_split = split_training_data(
        dataframe,
        target_column="SeriousDlqin2yrs",
        config=config,
    )

    assert first_split.train["Unnamed: 0"].tolist() == second_split.train["Unnamed: 0"].tolist()
    assert (
        first_split.validation["Unnamed: 0"].tolist()
        == second_split.validation["Unnamed: 0"].tolist()
    )
    assert first_split.test["Unnamed: 0"].tolist() == second_split.test["Unnamed: 0"].tolist()

    observed_ids = (
        set(first_split.train["Unnamed: 0"])
        | set(first_split.validation["Unnamed: 0"])
        | set(first_split.test["Unnamed: 0"])
    )
    assert len(first_split.train) == 28
    assert len(first_split.validation) == 6
    assert len(first_split.test) == 6
    assert observed_ids == set(dataframe["Unnamed: 0"])


def test_build_split_metadata_reports_partition_statistics() -> None:
    dataframe = _build_training_dataframe()
    partitions = split_training_data(
        dataframe,
        target_column="SeriousDlqin2yrs",
        config=SplitConfig(random_state=42),
    )

    metadata = build_split_metadata(
        partitions,
        target_column="SeriousDlqin2yrs",
        config=SplitConfig(random_state=42),
    )

    assert metadata["config"]["train_size"] == 0.70
    assert metadata["partitions"]["train"]["row_count"] == 28
    assert metadata["partitions"]["validation"]["row_count"] == 6
    assert metadata["partitions"]["test"]["row_count"] == 6
    assert metadata["partitions"]["train"]["positive_rate"] == pytest.approx(6 / 28)


def test_credit_risk_feature_builder_adds_missing_indicators_and_applies_clipping() -> None:
    dataframe = _build_training_dataframe().loc[:, [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ]]
    builder = CreditRiskFeatureBuilder()

    transformed = builder.fit_transform(dataframe)

    assert transformed.columns.tolist()[-2:] == [
        "MonthlyIncome_missing",
        "NumberOfDependents_missing",
    ]
    assert transformed.loc[0, "age"] == 18.0
    assert transformed.loc[0, "RevolvingUtilizationOfUnsecuredLines"] == 1.0
    assert transformed.loc[0, "DebtRatio"] == 5.0
    assert transformed.loc[0, "NumberOfTime30-59DaysPastDueNotWorse"] == 20.0
    assert transformed.loc[0, "NumberOfTimes90DaysLate"] == 20.0
    assert transformed.loc[0, "NumberOfTime60-89DaysPastDueNotWorse"] == 20.0
    assert transformed.loc[0, "MonthlyIncome_missing"] == 1
    assert transformed.loc[1, "MonthlyIncome_missing"] == 0
    assert transformed.loc[0, "NumberOfDependents_missing"] == 1


def test_credit_risk_feature_builder_raises_for_missing_required_columns() -> None:
    dataframe = _build_training_dataframe().drop(columns=["MonthlyIncome"])
    builder = CreditRiskFeatureBuilder()

    with pytest.raises(ValueError, match="missing required feature columns"):
        builder.fit(dataframe)


def test_create_preprocessing_pipeline_returns_expected_steps() -> None:
    pipeline = create_preprocessing_pipeline()

    assert list(pipeline.named_steps.keys()) == [
        "feature_builder",
        "imputer",
        "scaler",
    ]


def test_build_feature_schema_tracks_model_inputs_without_identifier_leakage() -> None:
    schema = build_feature_schema(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
    )

    assert "Unnamed: 0" not in schema.raw_feature_columns
    assert "SeriousDlqin2yrs" not in schema.raw_feature_columns
    assert schema.missing_indicator_features == (
        "MonthlyIncome_missing",
        "NumberOfDependents_missing",
    )
    assert "MonthlyIncome_missing" in schema.model_input_features
    assert "NumberOfDependents_missing" in schema.model_input_features
    assert schema.clipping_rules["age"] == {"lower": 18.0, "upper": 100.0}
    assert schema.to_dict()["schema_version"] == "1.0.0"
