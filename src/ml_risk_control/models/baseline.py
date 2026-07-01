"""Logistic-regression baseline model and artifact utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_risk_control.features.build import (
    DEFAULT_MISSING_INDICATOR_COLUMNS,
    FeatureSchema,
    build_feature_schema,
    create_preprocessing_pipeline,
)


@dataclass(frozen=True)
class LogisticRegressionBaselineConfig:
    """Configuration for the logistic-regression baseline classifier."""

    penalty: str | None = None
    solver: str = "lbfgs"
    C: float = 1.0
    max_iter: int = 1000
    class_weight: str | dict[int, float] | None = None
    random_state: int = 42
    fit_intercept: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration payload."""
        return asdict(self)

    def to_model_kwargs(self) -> dict[str, Any]:
        """Return estimator kwargs with deprecated defaults omitted."""
        params = self.to_dict()
        if params.get("penalty") is None:
            params.pop("penalty", None)
        return params


class LogisticRegressionBaseline:
    """Reusable logistic-regression baseline with shared preprocessing."""

    def __init__(
        self,
        *,
        id_column: str,
        target_column: str,
        model_version: str = "0.1.0",
        schema_version: str = "1.0.0",
        config: LogisticRegressionBaselineConfig | None = None,
        feature_schema: FeatureSchema | None = None,
        preprocessing_pipeline: Pipeline | None = None,
        missing_indicator_columns: tuple[str, ...] = DEFAULT_MISSING_INDICATOR_COLUMNS,
    ) -> None:
        self.id_column = id_column
        self.target_column = target_column
        self.model_version = model_version
        self.schema_version = schema_version
        self.config = config or LogisticRegressionBaselineConfig()
        self.missing_indicator_columns = missing_indicator_columns
        self.feature_schema = feature_schema or build_feature_schema(
            id_column=id_column,
            target_column=target_column,
            missing_indicator_columns=missing_indicator_columns,
            schema_version=schema_version,
        )
        self.preprocessing_pipeline = preprocessing_pipeline or create_preprocessing_pipeline(
            feature_columns=self.feature_schema.raw_feature_columns,
            missing_indicator_columns=missing_indicator_columns,
        )
        self.pipeline_: Pipeline | None = None
        self.training_summary_: dict[str, Any] | None = None

    def fit(self, dataframe: pd.DataFrame) -> LogisticRegressionBaseline:
        """Fit the shared preprocessing and logistic-regression classifier."""
        X, y = self._split_features_and_target(dataframe)
        classifier = LogisticRegression(**self.config.to_model_kwargs())
        self.pipeline_ = Pipeline(
            steps=[
                ("preprocessing", clone(self.preprocessing_pipeline)),
                ("classifier", classifier),
            ]
        )
        self.pipeline_.fit(X, y)
        self.training_summary_ = {
            "row_count": int(len(dataframe)),
            "positive_rate": float((y == 1).mean()),
            "trained_at_utc": datetime.now(UTC).isoformat(),
            "classifier_class": classifier.__class__.__name__,
        }
        return self

    def predict_proba(self, dataframe: pd.DataFrame) -> pd.Series:
        """Return positive-class probabilities indexed like the input dataframe."""
        pipeline = self._require_fitted_pipeline()
        features = self._select_feature_frame(dataframe)
        probabilities = pipeline.predict_proba(features)[:, 1]
        return pd.Series(probabilities, index=dataframe.index, name="predicted_probability")

    def predict(
        self,
        dataframe: pd.DataFrame,
        *,
        threshold: float = 0.5,
    ) -> pd.Series:
        """Return binary predictions at the provided probability threshold."""
        probabilities = self.predict_proba(dataframe)
        predictions = (probabilities >= threshold).astype(int)
        predictions.name = "predicted_label"
        return predictions

    def score_records(
        self,
        dataframe: pd.DataFrame,
        *,
        threshold: float = 0.5,
        include_identifier: bool = True,
    ) -> pd.DataFrame:
        """Return a scored dataframe with optional identifier passthrough."""
        scored = pd.DataFrame(index=dataframe.index)
        if include_identifier and self.id_column in dataframe.columns:
            scored[self.id_column] = dataframe[self.id_column].values

        scored["predicted_probability"] = self.predict_proba(dataframe)
        scored["predicted_label"] = (
            scored["predicted_probability"].ge(threshold).astype(int)
        )
        return scored

    def build_artifact_metadata(self) -> dict[str, Any]:
        """Return model metadata for artifact persistence and reporting."""
        return {
            "model_name": "logistic_regression_baseline",
            "model_version": self.model_version,
            "schema_version": self.feature_schema.schema_version,
            "id_column": self.id_column,
            "target_column": self.target_column,
            "raw_feature_columns": list(self.feature_schema.raw_feature_columns),
            "model_input_features": list(self.feature_schema.model_input_features),
            "missing_indicator_columns": list(self.missing_indicator_columns),
            "classifier_config": self.config.to_dict(),
            "pipeline_steps": self._pipeline_step_names(),
            "training_summary": self.training_summary_,
        }

    def save(self, path: Path) -> Path:
        """Persist the fitted baseline artifact bundle with joblib."""
        pipeline = self._require_fitted_pipeline()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": pipeline,
                "config": self.config.to_dict(),
                "feature_schema": self.feature_schema.to_dict(),
                "model_version": self.model_version,
                "schema_version": self.schema_version,
                "missing_indicator_columns": self.missing_indicator_columns,
                "training_summary": self.training_summary_,
                "artifact_metadata": self.build_artifact_metadata(),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> LogisticRegressionBaseline:
        """Load a persisted baseline artifact bundle from disk."""
        bundle = joblib.load(path)
        feature_schema = FeatureSchema(**bundle["feature_schema"])
        config = LogisticRegressionBaselineConfig(**bundle["config"])
        baseline = cls(
            id_column=feature_schema.id_column,
            target_column=feature_schema.target_column,
            model_version=bundle["model_version"],
            schema_version=bundle["schema_version"],
            config=config,
            feature_schema=feature_schema,
            preprocessing_pipeline=bundle["pipeline"].named_steps["preprocessing"],
            missing_indicator_columns=tuple(bundle["missing_indicator_columns"]),
        )
        baseline.pipeline_ = bundle["pipeline"]
        baseline.training_summary_ = bundle.get("training_summary")
        return baseline

    def _split_features_and_target(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if self.target_column not in dataframe.columns:
            msg = f"Target column '{self.target_column}' is missing from the training dataframe."
            raise ValueError(msg)

        feature_frame = self._select_feature_frame(dataframe)
        target = pd.to_numeric(dataframe[self.target_column], errors="raise")
        return feature_frame, target

    def _select_feature_frame(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        missing_columns = [
            column
            for column in self.feature_schema.raw_feature_columns
            if column not in dataframe.columns
        ]
        if missing_columns:
            msg = f"Input dataframe is missing model feature columns: {missing_columns}"
            raise ValueError(msg)
        return dataframe.loc[:, self.feature_schema.raw_feature_columns].copy()

    def _pipeline_step_names(self) -> list[str]:
        if self.pipeline_ is not None:
            return list(self.pipeline_.named_steps.keys())
        return ["preprocessing", "classifier"]

    def _require_fitted_pipeline(self) -> Pipeline:
        if self.pipeline_ is None:
            msg = "LogisticRegressionBaseline must be fitted or loaded before inference."
            raise ValueError(msg)
        return self.pipeline_
