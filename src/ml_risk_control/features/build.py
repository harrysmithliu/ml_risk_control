"""Shared feature splitting, preprocessing, and schema utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_risk_control.data.contracts import GMSC_FEATURE_COLUMNS

DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = GMSC_FEATURE_COLUMNS
DEFAULT_MISSING_INDICATOR_COLUMNS: tuple[str, ...] = ("MonthlyIncome", "NumberOfDependents")
DEFAULT_CLIPPING_RULES: dict[str, tuple[float | None, float | None]] = {
    "age": (18.0, 100.0),
    "RevolvingUtilizationOfUnsecuredLines": (0.0, 1.0),
    "DebtRatio": (0.0, 5.0),
    "NumberOfTime30-59DaysPastDueNotWorse": (0.0, 20.0),
    "NumberOfTime60-89DaysPastDueNotWorse": (0.0, 20.0),
    "NumberOfTimes90DaysLate": (0.0, 20.0),
}


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for reproducible stratified train/validation/test splitting."""

    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    stratify: bool = True

    def __post_init__(self) -> None:
        total = self.train_size + self.validation_size + self.test_size
        if abs(total - 1.0) > 1e-9:
            msg = "train_size + validation_size + test_size must equal 1.0."
            raise ValueError(msg)


@dataclass(frozen=True)
class DatasetPartitions:
    """Named training partitions for downstream modeling."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class FeatureSchema:
    """Versionable metadata describing model-facing features."""

    schema_version: str
    id_column: str
    target_column: str
    raw_feature_columns: tuple[str, ...]
    derived_feature_columns: tuple[str, ...]
    model_input_features: tuple[str, ...]
    missing_indicator_features: tuple[str, ...]
    clipping_rules: dict[str, dict[str, float | None]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the feature schema."""
        return asdict(self)


def _normalized_clipping_rules(
    rules: dict[str, tuple[float | None, float | None]],
) -> dict[str, dict[str, float | None]]:
    return {
        column: {"lower": bounds[0], "upper": bounds[1]}
        for column, bounds in rules.items()
    }


def _target_summary(series: pd.Series) -> dict[str, Any]:
    counts = series.value_counts(dropna=False).sort_index()
    non_missing = series.dropna()
    positive_rate = None
    if not non_missing.empty:
        positive_rate = float((non_missing == 1).mean())
    return {
        "row_count": int(len(series)),
        "non_null_count": int(non_missing.shape[0]),
        "positive_rate": positive_rate,
        "class_counts": {str(key): int(value) for key, value in counts.items()},
    }


def get_model_feature_columns(
    *,
    id_column: str,
    target_column: str,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
) -> tuple[str, ...]:
    """Return predictive feature columns with identifier and target excluded."""
    return tuple(
        column
        for column in feature_columns
        if column not in {id_column, target_column}
    )


def split_training_data(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    config: SplitConfig,
) -> DatasetPartitions:
    """Create reproducible train/validation/test partitions."""
    if target_column not in dataframe.columns:
        msg = f"Target column '{target_column}' is missing from the training dataframe."
        raise ValueError(msg)

    stratify_values = dataframe[target_column] if config.stratify else None

    train_validation, test = train_test_split(
        dataframe,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_values,
    )

    validation_share_within_remaining = config.validation_size / (
        config.train_size + config.validation_size
    )
    train_validation_stratify = (
        train_validation[target_column] if config.stratify else None
    )
    train, validation = train_test_split(
        train_validation,
        test_size=validation_share_within_remaining,
        random_state=config.random_state,
        stratify=train_validation_stratify,
    )

    return DatasetPartitions(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def build_split_metadata(
    partitions: DatasetPartitions,
    *,
    target_column: str,
    config: SplitConfig,
) -> dict[str, Any]:
    """Summarize split sizes and class balance for artifact export."""
    return {
        "config": asdict(config),
        "partitions": {
            "train": _target_summary(partitions.train[target_column]),
            "validation": _target_summary(partitions.validation[target_column]),
            "test": _target_summary(partitions.test[target_column]),
        },
    }


class CreditRiskFeatureBuilder(BaseEstimator, TransformerMixin):
    """Build bounded numeric features and explicit missingness indicators."""

    def __init__(
        self,
        *,
        feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
        missing_indicator_columns: tuple[str, ...] = DEFAULT_MISSING_INDICATOR_COLUMNS,
        clipping_rules: dict[str, tuple[float | None, float | None]] | None = None,
    ) -> None:
        self.feature_columns = feature_columns
        self.missing_indicator_columns = missing_indicator_columns
        self.clipping_rules = clipping_rules or DEFAULT_CLIPPING_RULES

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> CreditRiskFeatureBuilder:
        """Validate incoming columns and lock output feature order."""
        self._validate_input_columns(X)

        derived_columns = list(self.feature_columns)
        for column in self.missing_indicator_columns:
            if column in self.feature_columns:
                derived_columns.append(f"{column}_missing")

        self.output_feature_names_ = tuple(derived_columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a model-ready dataframe prior to imputation and scaling."""
        self._validate_input_columns(X)

        feature_frame = X.loc[:, self.feature_columns].copy()
        for column in self.feature_columns:
            feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")

        for column, (lower, upper) in self.clipping_rules.items():
            if column not in feature_frame.columns:
                continue
            feature_frame[column] = feature_frame[column].clip(lower=lower, upper=upper)

        for column in self.missing_indicator_columns:
            if column not in feature_frame.columns:
                continue
            feature_frame[f"{column}_missing"] = feature_frame[column].isna().astype(int)

        ordered_columns = list(getattr(self, "output_feature_names_", feature_frame.columns))
        return feature_frame.loc[:, ordered_columns]

    def get_feature_names_out(self, input_features: Any | None = None) -> list[str]:
        """Return deterministic output feature names."""
        del input_features
        if not hasattr(self, "output_feature_names_"):
            msg = "CreditRiskFeatureBuilder must be fitted before requesting feature names."
            raise ValueError(msg)
        return list(self.output_feature_names_)

    def _validate_input_columns(self, X: pd.DataFrame) -> None:
        missing_columns = [column for column in self.feature_columns if column not in X.columns]
        if missing_columns:
            msg = f"Input dataframe is missing required feature columns: {missing_columns}"
            raise ValueError(msg)


def create_preprocessing_pipeline(
    *,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    missing_indicator_columns: tuple[str, ...] = DEFAULT_MISSING_INDICATOR_COLUMNS,
    clipping_rules: dict[str, tuple[float | None, float | None]] | None = None,
) -> Pipeline:
    """Create the shared preprocessing pipeline for training and inference."""
    return Pipeline(
        steps=[
            (
                "feature_builder",
                CreditRiskFeatureBuilder(
                    feature_columns=feature_columns,
                    missing_indicator_columns=missing_indicator_columns,
                    clipping_rules=clipping_rules,
                ),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_feature_schema(
    *,
    id_column: str,
    target_column: str,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    missing_indicator_columns: tuple[str, ...] = DEFAULT_MISSING_INDICATOR_COLUMNS,
    clipping_rules: dict[str, tuple[float | None, float | None]] | None = None,
    schema_version: str = "1.0.0",
) -> FeatureSchema:
    """Construct feature-schema metadata from the shared preprocessing design."""
    effective_clipping_rules = clipping_rules or DEFAULT_CLIPPING_RULES
    model_feature_columns = get_model_feature_columns(
        id_column=id_column,
        target_column=target_column,
        feature_columns=feature_columns,
    )
    filtered_clipping_rules = {
        column: bounds
        for column, bounds in effective_clipping_rules.items()
        if column in model_feature_columns
    }
    derived_feature_columns = list(model_feature_columns)
    derived_feature_columns.extend(
        f"{column}_missing"
        for column in missing_indicator_columns
        if column in model_feature_columns
    )

    return FeatureSchema(
        schema_version=schema_version,
        id_column=id_column,
        target_column=target_column,
        raw_feature_columns=model_feature_columns,
        derived_feature_columns=tuple(derived_feature_columns),
        model_input_features=tuple(derived_feature_columns),
        missing_indicator_features=tuple(
            f"{column}_missing"
            for column in missing_indicator_columns
            if column in model_feature_columns
        ),
        clipping_rules=_normalized_clipping_rules(filtered_clipping_rules),
    )
