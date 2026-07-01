"""Local batch inference helpers for artifact-backed CSV-style scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from ml_risk_control.inference.service import (
    ApplicantValidationError,
    InferenceServiceError,
    LocalXGBoostInferenceService,
)


class BatchValidationError(InferenceServiceError):
    """Raised when an uploaded batch cannot be scored safely."""

    def __init__(
        self,
        message: str,
        *,
        file_errors: list[str] | None = None,
        row_errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.file_errors = file_errors or []
        self.row_errors = row_errors or []


@dataclass(frozen=True)
class BatchScoringSummary:
    """Compact summary of a scored applicant batch."""

    uploaded_row_count: int
    scored_row_count: int
    average_predicted_probability: float
    median_predicted_probability: float
    high_risk_row_count: int
    high_risk_share: float
    flagged_f1_threshold_count: int
    flagged_cost_threshold_count: int
    selected_candidate_source: str
    f1_threshold: float
    cost_threshold: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary payload."""
        return asdict(self)


@dataclass(frozen=True)
class BatchScoringResult:
    """Structured output for a successfully scored applicant batch."""

    scored_records: pd.DataFrame
    summary: BatchScoringSummary

    def to_dict(self) -> dict[str, Any]:
        """Return a summary-first payload with records rendered as dictionaries."""
        return {
            "summary": self.summary.to_dict(),
            "scored_records": self.scored_records.to_dict(orient="records"),
        }


class LocalXGBoostBatchInferenceService:
    """Artifact-backed batch inference service for local dataframe scoring."""

    def __init__(
        self,
        *,
        single_record_service: LocalXGBoostInferenceService | None = None,
    ) -> None:
        self.single_record_service = single_record_service or LocalXGBoostInferenceService()

    def load(self) -> LocalXGBoostBatchInferenceService:
        """Load the underlying single-record inference service."""
        self.single_record_service.load()
        return self

    @property
    def raw_feature_columns(self) -> tuple[str, ...]:
        """Return required raw feature columns for batch scoring."""
        return self.single_record_service.raw_feature_columns

    def validate_uploaded_dataframe(
        self,
        dataframe: pd.DataFrame,
        *,
        allow_additional_columns: bool = True,
    ) -> None:
        """Validate high-level batch structure before row-level scoring."""
        file_errors: list[str] = []

        if dataframe.empty:
            file_errors.append("Uploaded dataframe must contain at least one row.")

        duplicated_columns = dataframe.columns[dataframe.columns.duplicated()].tolist()
        if duplicated_columns:
            file_errors.append(f"Duplicate columns are not allowed: {duplicated_columns}")

        missing_required_columns = [
            column for column in self.raw_feature_columns if column not in dataframe.columns
        ]
        if missing_required_columns:
            file_errors.append(
                f"Missing required feature columns: {missing_required_columns}"
            )

        if not allow_additional_columns:
            unexpected_columns = sorted(
                set(dataframe.columns) - set(self.raw_feature_columns)
            )
            if unexpected_columns:
                file_errors.append(
                    f"Unexpected columns are not allowed in strict mode: {unexpected_columns}"
                )

        if file_errors:
            raise BatchValidationError(
                "Batch dataframe failed file-level validation.",
                file_errors=file_errors,
            )

    def score_dataframe(
        self,
        dataframe: pd.DataFrame,
        *,
        allow_additional_columns: bool = True,
    ) -> BatchScoringResult:
        """Score all rows in an uploaded dataframe and return enriched output."""
        self.validate_uploaded_dataframe(
            dataframe,
            allow_additional_columns=allow_additional_columns,
        )

        scored_rows: list[dict[str, Any]] = []
        row_errors: list[dict[str, Any]] = []

        for row_index, row in dataframe.iterrows():
            applicant_payload = {
                column: row[column]
                for column in self.raw_feature_columns
            }

            try:
                result = self.single_record_service.score_applicant(applicant_payload)
            except ApplicantValidationError as error:
                row_errors.append(
                    {
                        "row_index": int(row_index) if isinstance(row_index, int) else row_index,
                        "message": str(error),
                    }
                )
                continue

            threshold_lookup = {
                item.name: item for item in result.threshold_decisions
            }
            enriched_record = row.to_dict()
            enriched_record.update(
                {
                    "predicted_probability": result.predicted_probability,
                    "risk_band": result.risk_band,
                    "predicted_label_f1_threshold": threshold_lookup[
                        "f1_validation_threshold"
                    ].predicted_label,
                    "predicted_label_cost_threshold": threshold_lookup[
                        "cost_validation_threshold"
                    ].predicted_label,
                    "f1_threshold": threshold_lookup["f1_validation_threshold"].threshold,
                    "cost_threshold": threshold_lookup["cost_validation_threshold"].threshold,
                    "selected_candidate_source": result.selected_candidate_source,
                }
            )
            scored_rows.append(enriched_record)

        if row_errors:
            raise BatchValidationError(
                "Batch dataframe failed row-level validation.",
                row_errors=row_errors,
            )

        scored_records = pd.DataFrame(scored_rows, index=dataframe.index)
        summary = self._build_summary(scored_records)
        return BatchScoringResult(scored_records=scored_records, summary=summary)

    def _build_summary(self, scored_records: pd.DataFrame) -> BatchScoringSummary:
        if scored_records.empty:
            msg = "Cannot build a batch summary for an empty scored dataframe."
            raise ValueError(msg)

        first_row = scored_records.iloc[0]
        high_risk_count = int((scored_records["risk_band"] == "High").sum())

        return BatchScoringSummary(
            uploaded_row_count=int(len(scored_records)),
            scored_row_count=int(len(scored_records)),
            average_predicted_probability=float(
                scored_records["predicted_probability"].mean()
            ),
            median_predicted_probability=float(
                scored_records["predicted_probability"].median()
            ),
            high_risk_row_count=high_risk_count,
            high_risk_share=float(high_risk_count / len(scored_records)),
            flagged_f1_threshold_count=int(
                scored_records["predicted_label_f1_threshold"].sum()
            ),
            flagged_cost_threshold_count=int(
                scored_records["predicted_label_cost_threshold"].sum()
            ),
            selected_candidate_source=str(first_row["selected_candidate_source"]),
            f1_threshold=float(first_row["f1_threshold"]),
            cost_threshold=float(first_row["cost_threshold"]),
        )
