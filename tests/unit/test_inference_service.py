from __future__ import annotations

import math

import pytest

from ml_risk_control.inference.service import (
    ApplicantValidationError,
    LocalXGBoostInferenceService,
)


def _build_valid_applicant_payload() -> dict[str, float | int | None]:
    return {
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
    }


def test_local_xgboost_inference_service_loads_current_artifacts() -> None:
    service = LocalXGBoostInferenceService().load()

    snapshot = service.build_status_snapshot()

    assert snapshot["selected_candidate_source"] == "reference"
    assert snapshot["f1_threshold"] == pytest.approx(0.19962999820709218)
    assert snapshot["cost_threshold"] == pytest.approx(0.13704658508300782)
    assert snapshot["calibration_method"] == "sigmoid"
    assert len(snapshot["raw_feature_columns"]) == 10


def test_build_input_frame_supports_optional_missing_values_and_clipping() -> None:
    service = LocalXGBoostInferenceService().load()
    applicant = _build_valid_applicant_payload()
    applicant["MonthlyIncome"] = None
    applicant["NumberOfDependents"] = ""
    applicant["age"] = 120
    applicant["DebtRatio"] = 9.5

    frame = service.build_input_frame(applicant)

    assert frame.shape == (1, 10)
    assert frame.loc[0, "MonthlyIncome"] is None
    assert frame.loc[0, "NumberOfDependents"] is None
    assert frame.loc[0, "age"] == 100
    assert frame.loc[0, "DebtRatio"] == 5.0


def test_score_applicant_returns_probability_thresholds_and_risk_band() -> None:
    service = LocalXGBoostInferenceService().load()

    result = service.score_applicant(_build_valid_applicant_payload())

    assert math.isfinite(result.predicted_probability)
    assert 0.0 <= result.predicted_probability <= 1.0
    assert result.risk_band in {"Low", "Medium", "High"}
    assert result.primary_selection_metric == "average_precision"
    assert result.selected_candidate_source == "reference"
    assert len(result.threshold_decisions) == 2
    assert [decision.name for decision in result.threshold_decisions] == [
        "f1_validation_threshold",
        "cost_validation_threshold",
    ]
    assert result.calibration_summary["method"] == "sigmoid"
    assert result.calibration_summary["improved_validation_brier"] is False


def test_build_input_frame_rejects_negative_non_optional_values() -> None:
    service = LocalXGBoostInferenceService().load()
    applicant = _build_valid_applicant_payload()
    applicant["DebtRatio"] = -0.1

    with pytest.raises(ApplicantValidationError, match="DebtRatio"):
        service.build_input_frame(applicant)


def test_build_input_frame_rejects_non_integer_like_count_fields() -> None:
    service = LocalXGBoostInferenceService().load()
    applicant = _build_valid_applicant_payload()
    applicant["NumberOfTimes90DaysLate"] = 1.5

    with pytest.raises(ApplicantValidationError, match="integer-like"):
        service.build_input_frame(applicant)


def test_build_input_frame_rejects_unexpected_fields() -> None:
    service = LocalXGBoostInferenceService().load()
    applicant = _build_valid_applicant_payload()
    applicant["unexpected_column"] = 123

    with pytest.raises(ApplicantValidationError, match="Unexpected applicant fields"):
        service.build_input_frame(applicant)
