"""XGBoost champion-candidate model and artifact utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from ml_risk_control.features.build import (
    DEFAULT_MISSING_INDICATOR_COLUMNS,
    FeatureSchema,
    build_feature_schema,
    create_preprocessing_pipeline,
)


def _load_xgb_classifier() -> Any:
    """Load XGBClassifier lazily so package import survives optional runtime issues."""
    try:
        from xgboost import XGBClassifier
    except Exception as error:  # pragma: no cover - depends on runtime native library setup
        msg = (
            "xgboost is not available or its native library could not be loaded. "
            "Ensure the package is installed and the platform OpenMP runtime is available."
        )
        raise ImportError(msg) from error
    return XGBClassifier


@dataclass(frozen=True)
class XGBoostModelConfig:
    """Configuration for the XGBoost champion-candidate classifier."""

    objective: str = "binary:logistic"
    eval_metric: tuple[str, ...] = ("aucpr", "auc", "logloss")
    n_estimators: int = 600
    max_depth: int = 4
    min_child_weight: float = 5.0
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    max_delta_step: float = 0.0
    scale_pos_weight: float = 1.0
    booster: str = "gbtree"
    tree_method: str = "hist"
    device: str = "cpu"
    n_jobs: int = -1
    random_state: int = 42
    verbosity: int = 1
    early_stopping_rounds: int | None = 50
    importance_type: str = "gain"
    validate_parameters: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration payload."""
        return asdict(self)

    def to_model_kwargs(self) -> dict[str, Any]:
        """Return estimator kwargs for the local XGBoost sklearn API."""
        params = self.to_dict()
        params["eval_metric"] = list(self.eval_metric)
        return params


class XGBoostCreditRiskModel:
    """Reusable XGBoost classifier with shared preprocessing and artifact helpers."""

    def __init__(
        self,
        *,
        id_column: str,
        target_column: str,
        model_version: str = "0.1.0",
        schema_version: str = "1.0.0",
        config: XGBoostModelConfig | None = None,
        feature_schema: FeatureSchema | None = None,
        preprocessing_pipeline: Pipeline | None = None,
        missing_indicator_columns: tuple[str, ...] = DEFAULT_MISSING_INDICATOR_COLUMNS,
        native_importance_types: tuple[str, ...] = ("gain", "weight", "cover"),
    ) -> None:
        self.id_column = id_column
        self.target_column = target_column
        self.model_version = model_version
        self.schema_version = schema_version
        self.config = config or XGBoostModelConfig()
        self.missing_indicator_columns = missing_indicator_columns
        self.native_importance_types = native_importance_types
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
        self.preprocessing_pipeline_: Pipeline | None = None
        self.classifier_: Any | None = None
        self.training_summary_: dict[str, Any] | None = None
        self.evaluation_history_: dict[str, Any] | None = None
        self.transformed_feature_names_: tuple[str, ...] | None = None

    def fit(
        self,
        dataframe: pd.DataFrame,
        *,
        eval_dataframe: pd.DataFrame | None = None,
        verbose: bool = False,
    ) -> XGBoostCreditRiskModel:
        """Fit the shared preprocessing pipeline and XGBoost classifier."""
        X_train, y_train = self._split_features_and_target(dataframe)
        self.preprocessing_pipeline_ = clone(self.preprocessing_pipeline)
        train_matrix = self._fit_transform_features(self.preprocessing_pipeline_, X_train)
        classifier = self._build_classifier()

        eval_set = None
        eval_row_count = 0
        if eval_dataframe is not None:
            X_eval, y_eval = self._split_features_and_target(eval_dataframe)
            eval_matrix = self._transform_features(self.preprocessing_pipeline_, X_eval)
            eval_set = [(eval_matrix, y_eval)]
            eval_row_count = int(len(eval_dataframe))

        classifier.fit(
            train_matrix,
            y_train,
            eval_set=eval_set,
            verbose=verbose,
        )

        self.classifier_ = classifier
        self.evaluation_history_ = self._get_evaluation_history()
        self.training_summary_ = {
            "row_count": int(len(dataframe)),
            "positive_rate": float((y_train == 1).mean()),
            "eval_row_count": eval_row_count,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "classifier_class": classifier.__class__.__name__,
            "best_iteration": self._get_optional_attribute("best_iteration"),
            "best_score": self._get_optional_attribute("best_score"),
        }
        return self

    def predict_proba(self, dataframe: pd.DataFrame) -> pd.Series:
        """Return positive-class probabilities indexed like the input dataframe."""
        classifier = self._require_fitted_classifier()
        preprocessing_pipeline = self._require_fitted_preprocessing()
        feature_frame = self._select_feature_frame(dataframe)
        model_matrix = self._transform_features(preprocessing_pipeline, feature_frame)
        probabilities = classifier.predict_proba(model_matrix)[:, 1]
        return pd.Series(probabilities, index=dataframe.index, name="predicted_probability")

    def predict(
        self,
        dataframe: pd.DataFrame,
        *,
        threshold: float = 0.5,
    ) -> pd.Series:
        """Return binary predictions at the provided probability threshold."""
        probabilities = self.predict_proba(dataframe)
        predictions = probabilities.ge(threshold).astype(int)
        predictions.name = "predicted_label"
        return predictions

    def score_records(
        self,
        dataframe: pd.DataFrame,
        *,
        threshold: float = 0.5,
        include_identifier: bool = True,
    ) -> pd.DataFrame:
        """Return scored records with optional identifier passthrough."""
        scored = pd.DataFrame(index=dataframe.index)
        if include_identifier and self.id_column in dataframe.columns:
            scored[self.id_column] = dataframe[self.id_column].values

        scored["predicted_probability"] = self.predict_proba(dataframe)
        scored["predicted_label"] = scored["predicted_probability"].ge(threshold).astype(int)
        return scored

    def export_native_importance(self) -> dict[str, dict[str, float]]:
        """Return native XGBoost importance diagnostics keyed by business feature name."""
        classifier = self._require_fitted_classifier()
        feature_names = self._require_transformed_feature_names()
        booster = classifier.get_booster()

        results: dict[str, dict[str, float]] = {}
        for importance_type in self.native_importance_types:
            raw_scores = booster.get_score(importance_type=importance_type)
            ordered_scores = {
                feature_name: float(raw_scores.get(feature_name, 0.0))
                for feature_name in feature_names
            }
            results[importance_type] = dict(
                sorted(
                    ordered_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
        return results

    def build_artifact_metadata(self) -> dict[str, Any]:
        """Return model metadata for artifact persistence and reporting."""
        return {
            "model_name": "xgboost_credit_risk",
            "model_version": self.model_version,
            "schema_version": self.feature_schema.schema_version,
            "id_column": self.id_column,
            "target_column": self.target_column,
            "raw_feature_columns": list(self.feature_schema.raw_feature_columns),
            "model_input_features": list(self.feature_schema.model_input_features),
            "transformed_feature_names": list(self._require_transformed_feature_names()),
            "missing_indicator_columns": list(self.missing_indicator_columns),
            "classifier_config": self.config.to_dict(),
            "training_summary": self.training_summary_,
            "evaluation_history": self.evaluation_history_,
            "native_importance_types": list(self.native_importance_types),
            "native_importance": self.export_native_importance(),
        }

    def save(self, path: Path) -> Path:
        """Persist the fitted XGBoost artifact bundle with joblib."""
        preprocessing_pipeline = self._require_fitted_preprocessing()
        classifier = self._require_fitted_classifier()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "preprocessing_pipeline": preprocessing_pipeline,
                "classifier": classifier,
                "config": self.config.to_dict(),
                "feature_schema": self.feature_schema.to_dict(),
                "model_version": self.model_version,
                "schema_version": self.schema_version,
                "missing_indicator_columns": self.missing_indicator_columns,
                "native_importance_types": self.native_importance_types,
                "training_summary": self.training_summary_,
                "evaluation_history": self.evaluation_history_,
                "transformed_feature_names": self._require_transformed_feature_names(),
                "artifact_metadata": self.build_artifact_metadata(),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> XGBoostCreditRiskModel:
        """Load a persisted XGBoost artifact bundle from disk."""
        bundle = joblib.load(path)
        feature_schema = FeatureSchema(**bundle["feature_schema"])
        config = XGBoostModelConfig(**bundle["config"])
        model = cls(
            id_column=feature_schema.id_column,
            target_column=feature_schema.target_column,
            model_version=bundle["model_version"],
            schema_version=bundle["schema_version"],
            config=config,
            feature_schema=feature_schema,
            preprocessing_pipeline=bundle["preprocessing_pipeline"],
            missing_indicator_columns=tuple(bundle["missing_indicator_columns"]),
            native_importance_types=tuple(bundle["native_importance_types"]),
        )
        model.preprocessing_pipeline_ = bundle["preprocessing_pipeline"]
        model.classifier_ = bundle["classifier"]
        model.training_summary_ = bundle.get("training_summary")
        model.evaluation_history_ = bundle.get("evaluation_history")
        transformed_feature_names = bundle.get("transformed_feature_names")
        if transformed_feature_names is not None:
            model.transformed_feature_names_ = tuple(transformed_feature_names)
        return model

    def _build_classifier(self) -> Any:
        XGBClassifier = _load_xgb_classifier()
        return XGBClassifier(**self.config.to_model_kwargs())

    def _split_features_and_target(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if self.target_column not in dataframe.columns:
            msg = f"Target column '{self.target_column}' is missing from the training dataframe."
            raise ValueError(msg)

        feature_frame = self._select_feature_frame(dataframe)
        target = pd.to_numeric(dataframe[self.target_column], errors="raise").astype(int)
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

    def _fit_transform_features(
        self,
        preprocessing_pipeline: Pipeline,
        feature_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        transformed = preprocessing_pipeline.fit_transform(feature_frame)
        feature_names = self._extract_transformed_feature_names(preprocessing_pipeline)
        self.transformed_feature_names_ = tuple(feature_names)
        return pd.DataFrame(transformed, columns=feature_names, index=feature_frame.index)

    def _transform_features(
        self,
        preprocessing_pipeline: Pipeline,
        feature_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        transformed = preprocessing_pipeline.transform(feature_frame)
        feature_names = self._require_transformed_feature_names()
        return pd.DataFrame(transformed, columns=feature_names, index=feature_frame.index)

    def _extract_transformed_feature_names(self, preprocessing_pipeline: Pipeline) -> list[str]:
        feature_builder = preprocessing_pipeline.named_steps["feature_builder"]
        return feature_builder.get_feature_names_out()

    def _get_evaluation_history(self) -> dict[str, Any] | None:
        classifier = self._require_fitted_classifier()
        if hasattr(classifier, "evals_result"):
            return classifier.evals_result()
        return None

    def _get_optional_attribute(self, name: str) -> Any:
        classifier = self._require_fitted_classifier()
        return getattr(classifier, name, None)

    def _require_fitted_preprocessing(self) -> Pipeline:
        if self.preprocessing_pipeline_ is None:
            msg = "XGBoostCreditRiskModel must be fitted or loaded before inference."
            raise ValueError(msg)
        return self.preprocessing_pipeline_

    def _require_fitted_classifier(self) -> Any:
        if self.classifier_ is None:
            msg = "XGBoostCreditRiskModel must be fitted or loaded before inference."
            raise ValueError(msg)
        return self.classifier_

    def _require_transformed_feature_names(self) -> tuple[str, ...]:
        if self.transformed_feature_names_ is None:
            msg = "XGBoostCreditRiskModel must be fitted or loaded before feature export."
            raise ValueError(msg)
        return self.transformed_feature_names_
