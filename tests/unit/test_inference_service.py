from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
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


class _FakeXGBoostModel:
    def predict_proba(self, frame: pd.DataFrame) -> pd.Series:
        debt_ratio = float(frame.loc[0, "DebtRatio"])
        return pd.Series([min(max(debt_ratio / 4.0, 0.0), 1.0)])


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_loaded_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> LocalXGBoostInferenceService:
    artifact_dir = tmp_path / "artifacts" / "xgboost"
    artifact_dir.mkdir(parents=True)

    _write_json(
        artifact_dir / "feature_schema.json",
        {
            "feature_schema": {
                "raw_feature_columns": [
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
                ],
                "clipping_rules": {
                    "age": {"lower": 18, "upper": 100},
                    "DebtRatio": {"lower": 0.0, "upper": 5.0},
                },
            }
        },
    )
    _write_json(
        artifact_dir / "run_summary.json",
        {
            "selected_candidate_source": "reference",
        },
    )
    _write_json(
        artifact_dir / "threshold_selection_report.json",
        {
            "validation_selection": {
                "recommended_threshold": 0.19962999820709218,
            }
        },
    )
    _write_json(
        artifact_dir / "cost_analysis_report.json",
        {
            "validation_selection": {
                "recommended_threshold": 0.13704658508300782,
            }
        },
    )
    _write_json(
        artifact_dir / "calibration_report.json",
        {
            "calibration": {
                "method": "sigmoid",
                "strategy": "train_holdout",
            },
            "partitions": {
                "validation": {
                    "raw": {"brier_score": 0.051},
                    "calibrated": {"brier_score": 0.053},
                }
            },
        },
    )

    model_artifact_path = artifact_dir / "xgboost_credit_risk.joblib"
    model_artifact_path.write_bytes(b"fake-binary-artifact")

    monkeypatch.setattr(
        "ml_risk_control.inference.service.XGBoostCreditRiskModel.load",
        lambda path: _FakeXGBoostModel(),
    )

    return LocalXGBoostInferenceService(artifact_dir=artifact_dir).load()


@pytest.fixture
def loaded_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> LocalXGBoostInferenceService:
    return _build_loaded_service(tmp_path, monkeypatch)


def test_local_xgboost_inference_service_loads_current_artifacts(
    loaded_service: LocalXGBoostInferenceService,
) -> None:
    snapshot = loaded_service.build_status_snapshot()

    assert snapshot["selected_candidate_source"] == "reference"
    assert snapshot["f1_threshold"] == pytest.approx(0.19962999820709218)
    assert snapshot["cost_threshold"] == pytest.approx(0.13704658508300782)
    assert snapshot["calibration_method"] == "sigmoid"
    assert len(snapshot["raw_feature_columns"]) == 10


def test_build_input_frame_supports_optional_missing_values_and_clipping(
    loaded_service: LocalXGBoostInferenceService,
) -> None:
    applicant = _build_valid_applicant_payload()
    applicant["MonthlyIncome"] = None
    applicant["NumberOfDependents"] = ""
    applicant["age"] = 120
    applicant["DebtRatio"] = 9.5

    frame = loaded_service.build_input_frame(applicant)

    assert frame.shape == (1, 10)
    assert frame.loc[0, "MonthlyIncome"] is None
    assert frame.loc[0, "NumberOfDependents"] is None
    assert frame.loc[0, "age"] == 100
    assert frame.loc[0, "DebtRatio"] == 5.0


def test_score_applicant_returns_probability_thresholds_and_risk_band(
    loaded_service: LocalXGBoostInferenceService,
) -> None:
    result = loaded_service.score_applicant(_build_valid_applicant_payload())

    assert math.isfinite(result.predicted_probability)
    assert 0.0 <= result.predicted_probability <= 1.0
    assert result.predicted_probability == pytest.approx(0.08)
    assert result.risk_band == "Low"
    assert result.primary_selection_metric == "average_precision"
    assert result.selected_candidate_source == "reference"
    assert len(result.threshold_decisions) == 2
    assert [decision.name for decision in result.threshold_decisions] == [
        "f1_validation_threshold",
        "cost_validation_threshold",
    ]
    assert [decision.predicted_label for decision in result.threshold_decisions] == [0, 0]
    assert result.calibration_summary["method"] == "sigmoid"
    assert result.calibration_summary["improved_validation_brier"] is False


def test_build_input_frame_rejects_negative_non_optional_values(
    loaded_service: LocalXGBoostInferenceService,
) -> None:
    applicant = _build_valid_applicant_payload()
    applicant["DebtRatio"] = -0.1

    with pytest.raises(ApplicantValidationError, match="DebtRatio"):
        loaded_service.build_input_frame(applicant)


def test_build_input_frame_rejects_non_integer_like_count_fields(
    loaded_service: LocalXGBoostInferenceService,
) -> None:
    applicant = _build_valid_applicant_payload()
    applicant["NumberOfTimes90DaysLate"] = 1.5

    with pytest.raises(ApplicantValidationError, match="integer-like"):
        loaded_service.build_input_frame(applicant)


def test_build_input_frame_rejects_unexpected_fields(
    loaded_service: LocalXGBoostInferenceService,
) -> None:
    applicant = _build_valid_applicant_payload()
    applicant["unexpected_column"] = 123

    with pytest.raises(ApplicantValidationError, match="Unexpected applicant fields"):
        loaded_service.build_input_frame(applicant)
