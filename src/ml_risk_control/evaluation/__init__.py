"""Evaluation utilities for binary credit-risk models."""

from ml_risk_control.evaluation.comparison import (
    build_champion_decision_payload,
    build_default_comparison_payload,
    build_model_comparison_record,
    default_model_comparison_configs,
    rank_model_records,
    select_champion_record,
)
from ml_risk_control.evaluation.metrics import (
    build_confusion_matrix_payload,
    compute_ks_statistic,
    evaluate_binary_classifier,
)

__all__ = [
    "build_champion_decision_payload",
    "build_confusion_matrix_payload",
    "build_default_comparison_payload",
    "build_model_comparison_record",
    "compute_ks_statistic",
    "default_model_comparison_configs",
    "evaluate_binary_classifier",
    "rank_model_records",
    "select_champion_record",
]
