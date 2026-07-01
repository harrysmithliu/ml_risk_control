from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_risk_control.evaluation.comparison import (
    build_default_comparison_payload,
    build_model_comparison_record,
    default_model_comparison_configs,
    rank_model_records,
    select_champion_record,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics_payload(
    *,
    model_name: str,
    validation_ap: float,
    validation_roc: float,
    validation_ks: float,
    validation_brier: float,
    test_ap: float,
    test_roc: float,
    test_ks: float,
    test_brier: float,
) -> dict[str, object]:
    def partition(
        average_precision: float,
        roc_auc: float,
        ks_statistic: float,
        brier_score: float,
    ) -> dict[str, float]:
        return {
            "average_precision": average_precision,
            "roc_auc": roc_auc,
            "ks_statistic": ks_statistic,
            "brier_score": brier_score,
            "accuracy": 0.91,
            "precision": 0.42,
            "recall": 0.51,
            "f1": 0.46,
        }

    return {
        "model_name": model_name,
        "model_version": "0.1.0",
        "schema_version": "1.0.0",
        "partitions": {
            "validation": partition(
                validation_ap,
                validation_roc,
                validation_ks,
                validation_brier,
            ),
            "test": partition(
                test_ap,
                test_roc,
                test_ks,
                test_brier,
            ),
        },
    }


def _run_summary_payload(
    selected_candidate_source: str | None,
    *,
    classifier_class: str,
) -> dict[str, object]:
    return {
        "selected_candidate_source": selected_candidate_source,
        "training_summary": {
            "row_count": 105000,
            "positive_rate": 0.0668,
            "eval_row_count": 22500,
            "trained_at_utc": "2026-07-01T00:00:00+00:00",
            "classifier_class": classifier_class,
        },
    }


def _build_comparison_fixture(project_root: Path) -> None:
    artifact_root = project_root / "artifacts"

    _write_json(
        artifact_root / "baseline" / "baseline_metrics.json",
        _metrics_payload(
            model_name="logistic_regression_baseline",
            validation_ap=0.3224,
            validation_roc=0.8278,
            validation_ks=0.5130,
            validation_brier=0.0534,
            test_ap=0.3303,
            test_roc=0.8311,
            test_ks=0.5071,
            test_brier=0.0531,
        ),
    )
    _write_json(
        artifact_root / "baseline" / "run_summary.json",
        _run_summary_payload(
            None,
            classifier_class="LogisticRegression",
        ),
    )

    _write_json(
        artifact_root / "xgboost" / "xgboost_metrics.json",
        _metrics_payload(
            model_name="xgboost_credit_risk",
            validation_ap=0.4195,
            validation_roc=0.8711,
            validation_ks=0.5930,
            validation_brier=0.0485,
            test_ap=0.4122,
            test_roc=0.8717,
            test_ks=0.5899,
            test_brier=0.0485,
        ),
    )
    _write_json(
        artifact_root / "xgboost" / "run_summary.json",
        _run_summary_payload(
            "reference",
            classifier_class="XGBClassifier",
        ),
    )
    _write_json(
        artifact_root / "xgboost" / "threshold_selection_report.json",
        {
            "validation_selection": {
                "recommended_threshold": 0.19963,
            }
        },
    )
    _write_json(
        artifact_root / "xgboost" / "cost_analysis_report.json",
        {
            "validation_selection": {
                "recommended_threshold": 0.13705,
            }
        },
    )
    _write_json(
        artifact_root / "xgboost" / "calibration_report.json",
        {
            "calibration": {
                "method": "sigmoid",
            },
            "partitions": {
                "validation": {
                    "raw": {"brier_score": 0.0485},
                    "calibrated": {"brier_score": 0.0491},
                }
            },
        },
    )

    _write_json(
        artifact_root / "torch" / "torch_metrics.json",
        _metrics_payload(
            model_name="torch_mlp_challenger",
            validation_ap=0.3976,
            validation_roc=0.8723,
            validation_ks=0.5904,
            validation_brier=0.1494,
            test_ap=0.3959,
            test_roc=0.8692,
            test_ks=0.5847,
            test_brier=0.1494,
        ),
    )
    _write_json(
        artifact_root / "torch" / "run_summary.json",
        _run_summary_payload(
            None,
            classifier_class="TorchMLPClassifier",
        ),
    )


@pytest.fixture
def comparison_project_root(tmp_path: Path) -> Path:
    _build_comparison_fixture(tmp_path)
    return tmp_path


def test_default_model_comparison_configs_return_three_models() -> None:
    configs = default_model_comparison_configs(Path("/tmp/project"))

    assert [config.model_key for config in configs] == ["baseline", "xgboost", "torch"]
    assert any(config.designated_champion for config in configs)


def test_build_model_comparison_record_loads_xgboost_context(
    comparison_project_root: Path,
) -> None:
    xgboost_config = next(
        config
        for config in default_model_comparison_configs(comparison_project_root)
        if config.model_key == "xgboost"
    )

    record = build_model_comparison_record(xgboost_config)

    assert record["model_key"] == "xgboost"
    assert record["threshold_context"]["f1_threshold"] == pytest.approx(0.19963)
    assert record["threshold_context"]["cost_threshold"] == pytest.approx(0.13705)
    assert record["threshold_context"]["calibration_method"] == "sigmoid"
    assert record["threshold_context"]["calibration_improved_validation_brier"] is False
    assert record["interactive_inference_supported"] is True
    assert record["batch_inference_supported"] is True


def test_rank_model_records_prefers_highest_validation_average_precision(
    comparison_project_root: Path,
) -> None:
    records = [
        build_model_comparison_record(config)
        for config in default_model_comparison_configs(comparison_project_root)
    ]

    ranked = rank_model_records(records)

    assert [record["model_key"] for record in ranked] == ["xgboost", "torch", "baseline"]


def test_select_champion_record_keeps_xgboost_as_current_champion(
    comparison_project_root: Path,
) -> None:
    records = [
        build_model_comparison_record(config)
        for config in default_model_comparison_configs(comparison_project_root)
    ]

    champion = select_champion_record(records)

    assert champion["model_key"] == "xgboost"


def test_build_default_comparison_payload_contains_decision_and_table(
    comparison_project_root: Path,
) -> None:
    payload = build_default_comparison_payload(comparison_project_root)

    assert len(payload["model_records"]) == 3
    assert payload["decision"]["champion_model_key"] == "xgboost"
    assert len(payload["decision"]["comparison_table"]) == 3
    assert payload["decision"]["comparison_table"][0]["model_key"] == "xgboost"
