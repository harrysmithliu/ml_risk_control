from __future__ import annotations

from pathlib import Path

from ml_risk_control.evaluation.comparison import (
    build_default_comparison_payload,
    build_model_comparison_record,
    default_model_comparison_configs,
    rank_model_records,
    select_champion_record,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_default_model_comparison_configs_return_three_models() -> None:
    configs = default_model_comparison_configs(PROJECT_ROOT)

    assert [config.model_key for config in configs] == ["baseline", "xgboost", "torch"]
    assert any(config.designated_champion for config in configs)


def test_build_model_comparison_record_loads_xgboost_context() -> None:
    xgboost_config = next(
        config
        for config in default_model_comparison_configs(PROJECT_ROOT)
        if config.model_key == "xgboost"
    )

    record = build_model_comparison_record(xgboost_config)

    assert record["model_key"] == "xgboost"
    assert record["threshold_context"]["f1_threshold"] is not None
    assert record["threshold_context"]["cost_threshold"] is not None
    assert record["interactive_inference_supported"] is True
    assert record["batch_inference_supported"] is True


def test_rank_model_records_prefers_highest_validation_average_precision() -> None:
    records = [
        build_model_comparison_record(config)
        for config in default_model_comparison_configs(PROJECT_ROOT)
    ]

    ranked = rank_model_records(records)

    assert [record["model_key"] for record in ranked] == ["xgboost", "torch", "baseline"]


def test_select_champion_record_keeps_xgboost_as_current_champion() -> None:
    records = [
        build_model_comparison_record(config)
        for config in default_model_comparison_configs(PROJECT_ROOT)
    ]

    champion = select_champion_record(records)

    assert champion["model_key"] == "xgboost"


def test_build_default_comparison_payload_contains_decision_and_table() -> None:
    payload = build_default_comparison_payload(PROJECT_ROOT)

    assert len(payload["model_records"]) == 3
    assert payload["decision"]["champion_model_key"] == "xgboost"
    assert len(payload["decision"]["comparison_table"]) == 3
    assert payload["decision"]["comparison_table"][0]["model_key"] == "xgboost"
