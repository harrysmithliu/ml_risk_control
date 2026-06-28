from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_risk_control.data.contracts import RawDataContracts, build_gmsc_raw_data_contracts
from ml_risk_control.data.validation import (
    build_raw_validation_report,
    dataset_fingerprint,
    validate_exact_columns,
    validate_file_contract,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_contracts() -> RawDataContracts:
    return build_gmsc_raw_data_contracts(
        id_column="Unnamed: 0",
        target_column="SeriousDlqin2yrs",
        train_file_name="cs-training.csv",
        test_file_name="cs-test.csv",
    )


def _build_train_rows() -> list[dict[str, object]]:
    return [
        {
            "Unnamed: 0": 1,
            "SeriousDlqin2yrs": 0,
            "RevolvingUtilizationOfUnsecuredLines": 0.15,
            "age": 45,
            "NumberOfTime30-59DaysPastDueNotWorse": 0,
            "DebtRatio": 0.32,
            "MonthlyIncome": 5000,
            "NumberOfOpenCreditLinesAndLoans": 8,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60-89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 2,
        },
        {
            "Unnamed: 0": 2,
            "SeriousDlqin2yrs": 1,
            "RevolvingUtilizationOfUnsecuredLines": 0.91,
            "age": 38,
            "NumberOfTime30-59DaysPastDueNotWorse": 1,
            "DebtRatio": 0.67,
            "MonthlyIncome": 3200,
            "NumberOfOpenCreditLinesAndLoans": 5,
            "NumberOfTimes90DaysLate": 1,
            "NumberRealEstateLoansOrLines": 0,
            "NumberOfTime60-89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 1,
        },
    ]


def _build_test_rows() -> list[dict[str, object]]:
    return [
        {
            "Unnamed: 0": 3,
            "SeriousDlqin2yrs": None,
            "RevolvingUtilizationOfUnsecuredLines": 0.40,
            "age": 41,
            "NumberOfTime30-59DaysPastDueNotWorse": 0,
            "DebtRatio": 0.25,
            "MonthlyIncome": 6100,
            "NumberOfOpenCreditLinesAndLoans": 7,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60-89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 0,
        }
    ]


def _build_sample_submission_rows() -> list[dict[str, object]]:
    return [{"Id": 3, "Probability": 0.12}]


def test_validate_exact_columns_reports_optional_columns() -> None:
    result = validate_exact_columns(
        actual=["id", "target", "feature"],
        expected=["id", "feature"],
        optional=["target"],
    )

    assert result["missing_columns"] == []
    assert result["unexpected_columns"] == []
    assert result["optional_columns_present"] == ["target"]


def test_validate_file_contract_accepts_valid_training_file(tmp_path: Path) -> None:
    contracts = _build_contracts()
    train_path = tmp_path / "cs-training.csv"
    _write_csv(train_path, _build_train_rows())

    report = validate_file_contract(train_path, contracts.train)

    assert report["errors"] == []
    assert report["warnings"] == []
    assert report["row_count"] == 2
    assert report["id_is_unique"] is True
    assert report["target_is_binary"] is True
    assert report["target_summary"]["counts"] == {"0": 1, "1": 1}
    assert report["data_quality_flags"]["age_below_18_count"] == 0
    assert report["data_quality_flags"]["debt_ratio_above_5_count"] == 0


def test_validate_file_contract_adds_warning_level_data_quality_flags(tmp_path: Path) -> None:
    contracts = _build_contracts()
    train_path = tmp_path / "cs-training.csv"
    rows = _build_train_rows()
    rows[0]["age"] = 17
    rows[0]["DebtRatio"] = 6.5
    rows[0]["RevolvingUtilizationOfUnsecuredLines"] = 1.4
    rows[0]["MonthlyIncome"] = 0
    rows[0]["NumberOfTime30-59DaysPastDueNotWorse"] = 30
    rows[0]["NumberOfTimes90DaysLate"] = 25
    _write_csv(train_path, rows)

    report = validate_file_contract(train_path, contracts.train)

    assert report["errors"] == []
    assert report["data_quality_flags"]["age_below_18_count"] == 1
    assert report["data_quality_flags"]["debt_ratio_above_5_count"] == 1
    assert report["data_quality_flags"]["revolving_utilization_above_1_count"] == 1
    assert report["data_quality_flags"]["monthly_income_non_positive_count"] == 1
    assert report["data_quality_flags"]["delinquency_count_above_20"][
        "NumberOfTime30-59DaysPastDueNotWorse"
    ] == 1
    assert report["data_quality_flags"]["delinquency_count_above_20"][
        "NumberOfTimes90DaysLate"
    ] == 1
    assert "Detected 1 rows with age below 18." in report["warnings"]
    assert "Detected rows where debt ratio exceeds 5.0." in report["warnings"]
    assert "Detected rows where revolving utilization exceeds 1.0." in report["warnings"]
    assert "Detected rows with non-positive monthly income values." in report["warnings"]


def test_validate_file_contract_rejects_unlabeled_file_with_populated_target(tmp_path: Path) -> None:
    contracts = _build_contracts()
    test_path = tmp_path / "cs-test.csv"
    rows = _build_test_rows()
    rows[0]["SeriousDlqin2yrs"] = 1
    _write_csv(test_path, rows)

    report = validate_file_contract(test_path, contracts.test)

    assert "Target column contains values in a file that should be unlabeled." in report["errors"]


def test_build_raw_validation_report_handles_optional_missing_sample_submission(
    tmp_path: Path,
) -> None:
    contracts = _build_contracts()
    _write_csv(tmp_path / "cs-training.csv", _build_train_rows())
    _write_csv(tmp_path / "cs-test.csv", _build_test_rows())

    report = build_raw_validation_report(
        raw_data_dir=tmp_path,
        backend="local",
        contracts=contracts,
    )

    assert report["status"] == "passed"
    assert report["dataset_fingerprint"] is not None
    assert report["files"]["sample_submission"]["errors"] == []
    assert report["files"]["sample_submission"]["warnings"] == ["File does not exist."]


def test_build_raw_validation_report_includes_all_files_when_present(tmp_path: Path) -> None:
    contracts = _build_contracts()
    _write_csv(tmp_path / "cs-training.csv", _build_train_rows())
    _write_csv(tmp_path / "cs-test.csv", _build_test_rows())
    _write_csv(tmp_path / "sampleEntry.csv", _build_sample_submission_rows())

    report = build_raw_validation_report(
        raw_data_dir=tmp_path,
        backend="local",
        contracts=contracts,
    )

    assert report["status"] == "passed"
    assert sorted(report["files"].keys()) == ["sample_submission", "test", "train"]
    assert (
        report["files"]["test"]["warnings"]
        == ["Target placeholder column is present in the unlabeled test file."]
    )
    assert report["files"]["sample_submission"]["errors"] == []
    assert report["dataset_fingerprint"] == dataset_fingerprint(
        {
            name: file_report["sha256"]
            for name, file_report in report["files"].items()
            if file_report.get("sha256")
        }
    )
