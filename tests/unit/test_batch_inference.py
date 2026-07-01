from __future__ import annotations

import pandas as pd
import pytest

from ml_risk_control.inference.batch import (
    BatchValidationError,
    LocalXGBoostBatchInferenceService,
)
from ml_risk_control.inference.service import (
    ApplicantScoreResult,
    ApplicantValidationError,
    ThresholdDecision,
)

RAW_FEATURE_COLUMNS: tuple[str, ...] = (
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
)


def _build_valid_batch_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "RevolvingUtilizationOfUnsecuredLines": 0.45,
                "age": 42,
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.32,
                "MonthlyIncome": 6500,
                "NumberOfOpenCreditLinesAndLoans": 8,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2,
                "customer_ref": "A001",
            },
            {
                "RevolvingUtilizationOfUnsecuredLines": 0.80,
                "age": 60,
                "NumberOfTime30-59DaysPastDueNotWorse": 3,
                "DebtRatio": 0.52,
                "MonthlyIncome": 6500,
                "NumberOfOpenCreditLinesAndLoans": 8,
                "NumberOfTimes90DaysLate": 1,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2,
                "customer_ref": "A002",
            },
        ]
    )


class FakeSingleRecordService:
    def __init__(self) -> None:
        self.raw_feature_columns = RAW_FEATURE_COLUMNS
        self.loaded = False

    def load(self) -> FakeSingleRecordService:
        self.loaded = True
        return self

    def score_applicant(self, applicant: dict[str, object]) -> ApplicantScoreResult:
        delinquency_count = int(applicant["NumberOfTimes90DaysLate"])
        probability = 0.05 if delinquency_count == 0 else 0.35
        risk_band = "Low" if probability < 0.10 else "High"
        return ApplicantScoreResult(
            predicted_probability=probability,
            risk_band=risk_band,
            threshold_decisions=(
                ThresholdDecision(
                    name="f1_validation_threshold",
                    threshold=0.20,
                    predicted_label=int(probability >= 0.20),
                ),
                ThresholdDecision(
                    name="cost_validation_threshold",
                    threshold=0.14,
                    predicted_label=int(probability >= 0.14),
                ),
            ),
            selected_candidate_source="reference",
            primary_selection_metric="average_precision",
            calibration_summary={
                "available": True,
                "method": "sigmoid",
                "strategy": "train_holdout",
                "validation_brier_raw": 0.05,
                "validation_brier_calibrated": 0.051,
                "improved_validation_brier": False,
            },
            input_frame=applicant,
        )


class FakeFailingSingleRecordService(FakeSingleRecordService):
    def score_applicant(self, applicant: dict[str, object]) -> ApplicantScoreResult:
        if int(applicant["NumberOfTimes90DaysLate"]) > 0:
            msg = "Field 'NumberOfTimes90DaysLate' must be integer-like."
            raise ApplicantValidationError(msg)
        return super().score_applicant(applicant)


def test_batch_inference_service_load_delegates_to_single_record_service() -> None:
    single_record_service = FakeSingleRecordService()
    service = LocalXGBoostBatchInferenceService(
        single_record_service=single_record_service
    )

    result = service.load()

    assert result is service
    assert single_record_service.loaded is True


def test_validate_uploaded_dataframe_rejects_missing_required_columns() -> None:
    service = LocalXGBoostBatchInferenceService(
        single_record_service=FakeSingleRecordService()
    )
    dataframe = _build_valid_batch_dataframe().drop(columns=["DebtRatio"])

    with pytest.raises(BatchValidationError) as exc_info:
        service.validate_uploaded_dataframe(dataframe)

    assert "Missing required feature columns" in exc_info.value.file_errors[0]


def test_validate_uploaded_dataframe_rejects_additional_columns_in_strict_mode() -> None:
    service = LocalXGBoostBatchInferenceService(
        single_record_service=FakeSingleRecordService()
    )
    dataframe = _build_valid_batch_dataframe()

    with pytest.raises(BatchValidationError) as exc_info:
        service.validate_uploaded_dataframe(
            dataframe,
            allow_additional_columns=False,
        )

    assert "Unexpected columns are not allowed in strict mode" in exc_info.value.file_errors[0]


def test_score_dataframe_returns_enriched_records_and_summary() -> None:
    service = LocalXGBoostBatchInferenceService(
        single_record_service=FakeSingleRecordService()
    )
    dataframe = _build_valid_batch_dataframe()

    result = service.score_dataframe(dataframe)

    assert result.scored_records["customer_ref"].tolist() == ["A001", "A002"]
    assert result.scored_records["predicted_probability"].tolist() == pytest.approx(
        [0.05, 0.35]
    )
    assert result.scored_records["risk_band"].tolist() == ["Low", "High"]
    assert result.scored_records["predicted_label_f1_threshold"].tolist() == [0, 1]
    assert result.scored_records["predicted_label_cost_threshold"].tolist() == [0, 1]
    assert result.summary.uploaded_row_count == 2
    assert result.summary.scored_row_count == 2
    assert result.summary.average_predicted_probability == pytest.approx(0.20)
    assert result.summary.median_predicted_probability == pytest.approx(0.20)
    assert result.summary.high_risk_row_count == 1
    assert result.summary.high_risk_share == pytest.approx(0.5)
    assert result.summary.flagged_f1_threshold_count == 1
    assert result.summary.flagged_cost_threshold_count == 1
    assert result.summary.selected_candidate_source == "reference"
    assert result.summary.f1_threshold == pytest.approx(0.20)
    assert result.summary.cost_threshold == pytest.approx(0.14)


def test_score_dataframe_fails_closed_on_row_level_validation_errors() -> None:
    service = LocalXGBoostBatchInferenceService(
        single_record_service=FakeFailingSingleRecordService()
    )
    dataframe = _build_valid_batch_dataframe()

    with pytest.raises(BatchValidationError) as exc_info:
        service.score_dataframe(dataframe)

    assert exc_info.value.file_errors == []
    assert len(exc_info.value.row_errors) == 1
    assert exc_info.value.row_errors[0]["row_index"] == 1
    assert "integer-like" in exc_info.value.row_errors[0]["message"]
