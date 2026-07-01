"""Local inference services for artifact-backed single-applicant scoring."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml_risk_control.config import get_settings
from ml_risk_control.models.xgboost_model import XGBoostCreditRiskModel

DEFAULT_XGBOOST_ARTIFACT_DIR = Path("artifacts") / "xgboost"
DEFAULT_MODEL_ARTIFACT_NAME = "xgboost_credit_risk.joblib"
OPTIONAL_INPUT_FIELDS: tuple[str, ...] = ("MonthlyIncome", "NumberOfDependents")
INTEGER_LIKE_FIELDS: tuple[str, ...] = (
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
)
NON_NEGATIVE_FIELDS: tuple[str, ...] = (
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
)


class InferenceServiceError(RuntimeError):
    """Base error for local inference service failures."""


class ArtifactLoadError(InferenceServiceError):
    """Raised when the local model bundle or required JSON artifacts are unavailable."""


class ApplicantValidationError(InferenceServiceError):
    """Raised when an applicant payload cannot be normalized into model input."""


@dataclass(frozen=True)
class ThresholdDecision:
    """Decision outcome for a single saved probability threshold."""

    name: str
    threshold: float
    predicted_label: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable threshold decision payload."""
        return asdict(self)


@dataclass(frozen=True)
class ApplicantScoreResult:
    """Structured single-applicant scoring payload for the Streamlit layer."""

    predicted_probability: float
    risk_band: str
    threshold_decisions: tuple[ThresholdDecision, ...]
    selected_candidate_source: str
    primary_selection_metric: str
    calibration_summary: dict[str, Any]
    input_frame: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result payload."""
        payload = asdict(self)
        payload["threshold_decisions"] = [
            decision.to_dict() for decision in self.threshold_decisions
        ]
        return payload


class LocalXGBoostInferenceService:
    """Artifact-backed local inference service for single-record applicant scoring."""

    def __init__(
        self,
        *,
        artifact_dir: Path | None = None,
        model_artifact_name: str = DEFAULT_MODEL_ARTIFACT_NAME,
        low_risk_cutoff: float = 0.10,
    ) -> None:
        settings = get_settings()
        self.project_root = settings.project_root
        self.artifact_dir = (
            artifact_dir
            if artifact_dir is not None
            else self.project_root / DEFAULT_XGBOOST_ARTIFACT_DIR
        )
        self.model_artifact_path = self.artifact_dir / model_artifact_name
        self.low_risk_cutoff = low_risk_cutoff

        self._model: XGBoostCreditRiskModel | None = None
        self._feature_schema: dict[str, Any] | None = None
        self._run_summary: dict[str, Any] | None = None
        self._threshold_selection_report: dict[str, Any] | None = None
        self._cost_analysis_report: dict[str, Any] | None = None
        self._calibration_report: dict[str, Any] | None = None

    def load(self) -> LocalXGBoostInferenceService:
        """Load the persisted model bundle and supporting metadata into memory."""
        self._model = XGBoostCreditRiskModel.load(self.model_artifact_path)
        self._feature_schema = self._read_json("feature_schema.json")
        self._run_summary = self._read_json("run_summary.json")
        self._threshold_selection_report = self._read_json("threshold_selection_report.json")
        self._cost_analysis_report = self._read_json("cost_analysis_report.json")
        self._calibration_report = self._read_json("calibration_report.json")
        return self

    def score_applicant(
        self,
        applicant: Mapping[str, Any],
    ) -> ApplicantScoreResult:
        """Score a single applicant payload against the persisted XGBoost artifact."""
        model = self._require_model()
        normalized_frame = self.build_input_frame(applicant)
        probability = float(model.predict_proba(normalized_frame).iloc[0])

        f1_threshold = self._threshold_selection_payload()["validation_selection"][
            "recommended_threshold"
        ]
        cost_threshold = self._cost_analysis_payload()["validation_selection"][
            "recommended_threshold"
        ]

        threshold_decisions = (
            ThresholdDecision(
                name="f1_validation_threshold",
                threshold=float(f1_threshold),
                predicted_label=int(probability >= float(f1_threshold)),
            ),
            ThresholdDecision(
                name="cost_validation_threshold",
                threshold=float(cost_threshold),
                predicted_label=int(probability >= float(cost_threshold)),
            ),
        )

        return ApplicantScoreResult(
            predicted_probability=probability,
            risk_band=self._build_risk_band(probability, f1_threshold=float(f1_threshold)),
            threshold_decisions=threshold_decisions,
            selected_candidate_source=str(
                self._run_summary_payload().get("selected_candidate_source", "unknown")
            ),
            primary_selection_metric="average_precision",
            calibration_summary=self._build_calibration_summary(),
            input_frame=normalized_frame.iloc[0].to_dict(),
        )

    def build_input_frame(
        self,
        applicant: Mapping[str, Any],
    ) -> pd.DataFrame:
        """Normalize a single applicant payload into the model's raw feature frame."""
        raw_feature_columns = self.raw_feature_columns
        clipping_rules = self.clipping_rules
        normalized_record: dict[str, Any] = {}

        unexpected_fields = sorted(set(applicant) - set(raw_feature_columns))
        if unexpected_fields:
            msg = f"Unexpected applicant fields: {unexpected_fields}"
            raise ApplicantValidationError(msg)

        for column in raw_feature_columns:
            value = applicant.get(column)
            normalized_record[column] = self._normalize_input_value(
                column=column,
                value=value,
                clipping_rules=clipping_rules,
            )

        return pd.DataFrame([normalized_record], columns=list(raw_feature_columns))

    def build_status_snapshot(self) -> dict[str, Any]:
        """Return a UI-friendly summary of the currently loaded artifact bundle."""
        threshold_payload = self._threshold_selection_payload()
        cost_payload = self._cost_analysis_payload()
        calibration_payload = self._calibration_payload()

        return {
            "artifact_dir": str(self.artifact_dir),
            "model_artifact_path": str(self.model_artifact_path),
            "selected_candidate_source": self._run_summary_payload().get(
                "selected_candidate_source",
                "unknown",
            ),
            "f1_threshold": threshold_payload["validation_selection"]["recommended_threshold"],
            "cost_threshold": cost_payload["validation_selection"]["recommended_threshold"],
            "calibration_method": calibration_payload["calibration"].get("method"),
            "calibration_strategy": calibration_payload["calibration"].get("strategy"),
            "raw_feature_columns": list(self.raw_feature_columns),
        }

    @property
    def raw_feature_columns(self) -> tuple[str, ...]:
        """Return the current raw feature columns expected by the model artifact."""
        payload = self._feature_schema_payload()["feature_schema"]
        return tuple(payload["raw_feature_columns"])

    @property
    def clipping_rules(self) -> dict[str, dict[str, float | None]]:
        """Return per-column clipping rules from the saved feature schema."""
        payload = self._feature_schema_payload()["feature_schema"]
        return payload.get("clipping_rules", {})

    def _normalize_input_value(
        self,
        *,
        column: str,
        value: Any,
        clipping_rules: dict[str, dict[str, float | None]],
    ) -> Any:
        if self._is_null_like(value):
            if column in OPTIONAL_INPUT_FIELDS:
                return None
            msg = f"Field '{column}' is required."
            raise ApplicantValidationError(msg)

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            msg = f"Field '{column}' must be numeric."
            raise ApplicantValidationError(msg) from error

        if column == "age":
            if numeric_value < 18:
                msg = "Field 'age' must be at least 18."
                raise ApplicantValidationError(msg)
        elif column in NON_NEGATIVE_FIELDS and numeric_value < 0:
            msg = f"Field '{column}' must be non-negative."
            raise ApplicantValidationError(msg)

        if column in INTEGER_LIKE_FIELDS:
            rounded_value = round(numeric_value)
            if abs(numeric_value - rounded_value) > 1e-9:
                msg = f"Field '{column}' must be integer-like."
                raise ApplicantValidationError(msg)
            numeric_value = float(rounded_value)

        bounds = clipping_rules.get(column, {})
        lower = bounds.get("lower")
        upper = bounds.get("upper")
        if lower is not None:
            numeric_value = max(numeric_value, float(lower))
        if upper is not None:
            numeric_value = min(numeric_value, float(upper))

        if column in INTEGER_LIKE_FIELDS:
            return int(round(numeric_value))
        return numeric_value

    def _build_risk_band(self, probability: float, *, f1_threshold: float) -> str:
        if probability < self.low_risk_cutoff:
            return "Low"
        if probability < f1_threshold:
            return "Medium"
        return "High"

    def _build_calibration_summary(self) -> dict[str, Any]:
        calibration_payload = self._calibration_payload()
        validation_partition = calibration_payload.get("partitions", {}).get("validation", {})
        raw_metrics = validation_partition.get("raw", {})
        calibrated_metrics = validation_partition.get("calibrated", {})
        raw_brier = raw_metrics.get("brier_score")
        calibrated_brier = calibrated_metrics.get("brier_score")
        improved_brier = None
        if raw_brier is not None and calibrated_brier is not None:
            improved_brier = float(calibrated_brier) < float(raw_brier)

        return {
            "available": True,
            "method": calibration_payload.get("calibration", {}).get("method"),
            "strategy": calibration_payload.get("calibration", {}).get("strategy"),
            "validation_brier_raw": raw_brier,
            "validation_brier_calibrated": calibrated_brier,
            "improved_validation_brier": improved_brier,
        }

    def _read_json(self, file_name: str) -> dict[str, Any]:
        path = self.artifact_dir / file_name
        if not path.exists():
            msg = f"Required artifact file is missing: {path}"
            raise ArtifactLoadError(msg)
        return json.loads(path.read_text(encoding="utf-8"))

    def _feature_schema_payload(self) -> dict[str, Any]:
        if self._feature_schema is None:
            msg = "Inference service has not been loaded."
            raise ArtifactLoadError(msg)
        return self._feature_schema

    def _run_summary_payload(self) -> dict[str, Any]:
        if self._run_summary is None:
            msg = "Inference service has not been loaded."
            raise ArtifactLoadError(msg)
        return self._run_summary

    def _threshold_selection_payload(self) -> dict[str, Any]:
        if self._threshold_selection_report is None:
            msg = "Inference service has not been loaded."
            raise ArtifactLoadError(msg)
        return self._threshold_selection_report

    def _cost_analysis_payload(self) -> dict[str, Any]:
        if self._cost_analysis_report is None:
            msg = "Inference service has not been loaded."
            raise ArtifactLoadError(msg)
        return self._cost_analysis_report

    def _calibration_payload(self) -> dict[str, Any]:
        if self._calibration_report is None:
            msg = "Inference service has not been loaded."
            raise ArtifactLoadError(msg)
        return self._calibration_report

    def _require_model(self) -> XGBoostCreditRiskModel:
        if self._model is None:
            msg = "Inference service has not been loaded."
            raise ArtifactLoadError(msg)
        return self._model

    @staticmethod
    def _is_null_like(value: Any) -> bool:
        return value is None or (isinstance(value, str) and value.strip() == "")
