"""Model-comparison and champion-selection utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelComparisonConfig:
    """Static configuration for a persisted model candidate."""

    model_key: str
    label: str
    model_family: str
    artifact_dir: Path
    metrics_file: str
    run_summary_file: str
    threshold_selection_file: str | None = None
    cost_analysis_file: str | None = None
    calibration_file: str | None = None
    interactive_inference_supported: bool = False
    batch_inference_supported: bool = False
    designated_champion: bool = False


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return _read_json(path)


def _extract_partition_metrics(
    metrics_payload: dict[str, Any],
    partition_name: str,
) -> dict[str, float]:
    partition = metrics_payload["partitions"][partition_name]
    return {
        "average_precision": float(partition["average_precision"]),
        "roc_auc": float(partition["roc_auc"]),
        "ks_statistic": float(partition["ks_statistic"]),
        "brier_score": float(partition["brier_score"]),
        "accuracy": float(partition["accuracy"]),
        "precision": float(partition["precision"]),
        "recall": float(partition["recall"]),
        "f1": float(partition["f1"]),
    }


def _extract_threshold_context(
    threshold_payload: dict[str, Any] | None,
    cost_payload: dict[str, Any] | None,
    calibration_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "f1_threshold": None,
        "cost_threshold": None,
        "calibration_method": None,
        "calibration_improved_validation_brier": None,
    }
    if threshold_payload is not None:
        context["f1_threshold"] = float(
            threshold_payload["validation_selection"]["recommended_threshold"]
        )
    if cost_payload is not None:
        context["cost_threshold"] = float(
            cost_payload["validation_selection"]["recommended_threshold"]
        )
    if calibration_payload is not None:
        context["calibration_method"] = calibration_payload["calibration"].get("method")
        validation_raw = calibration_payload["partitions"]["validation"]["raw"]["brier_score"]
        validation_calibrated = calibration_payload["partitions"]["validation"]["calibrated"][
            "brier_score"
        ]
        context["calibration_improved_validation_brier"] = bool(
            validation_calibrated < validation_raw
        )
    return context


def build_model_comparison_record(config: ModelComparisonConfig) -> dict[str, Any]:
    """Load persisted artifacts for one model and normalize a comparison record."""
    metrics_payload = _read_json(config.artifact_dir / config.metrics_file)
    run_summary_payload = _read_json(config.artifact_dir / config.run_summary_file)
    threshold_payload = _read_optional_json(
        None
        if config.threshold_selection_file is None
        else config.artifact_dir / config.threshold_selection_file
    )
    cost_payload = _read_optional_json(
        None
        if config.cost_analysis_file is None
        else config.artifact_dir / config.cost_analysis_file
    )
    calibration_payload = _read_optional_json(
        None if config.calibration_file is None else config.artifact_dir / config.calibration_file
    )

    training_summary = run_summary_payload.get("training_summary", {})
    threshold_context = _extract_threshold_context(
        threshold_payload,
        cost_payload,
        calibration_payload,
    )

    return {
        "model_key": config.model_key,
        "label": config.label,
        "model_family": config.model_family,
        "model_name": metrics_payload["model_name"],
        "model_version": metrics_payload["model_version"],
        "schema_version": metrics_payload["schema_version"],
        "designated_champion": config.designated_champion,
        "interactive_inference_supported": config.interactive_inference_supported,
        "batch_inference_supported": config.batch_inference_supported,
        "selected_candidate_source": run_summary_payload.get("selected_candidate_source"),
        "validation_metrics": _extract_partition_metrics(metrics_payload, "validation"),
        "test_metrics": _extract_partition_metrics(metrics_payload, "test"),
        "threshold_context": threshold_context,
        "training_summary": {
            "row_count": training_summary.get("row_count"),
            "positive_rate": training_summary.get("positive_rate"),
            "eval_row_count": training_summary.get("eval_row_count"),
            "trained_at_utc": training_summary.get("trained_at_utc"),
            "classifier_class": training_summary.get("classifier_class"),
        },
    }


def rank_model_records(model_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return records sorted by validation PR-AUC, then test PR-AUC, then test ROC-AUC."""
    return sorted(
        model_records,
        key=lambda item: (
            item["validation_metrics"]["average_precision"],
            item["test_metrics"]["average_precision"],
            item["test_metrics"]["roc_auc"],
        ),
        reverse=True,
    )


def select_champion_record(model_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the current PoC champion-selection rule to the comparison records."""
    ranked_records = rank_model_records(model_records)
    if not ranked_records:
        msg = "Champion selection requires at least one model record."
        raise ValueError(msg)

    best_record = ranked_records[0]
    designated = next((record for record in ranked_records if record["designated_champion"]), None)
    if designated is None:
        return best_record

    best_test_ap = best_record["test_metrics"]["average_precision"]
    designated_test_ap = designated["test_metrics"]["average_precision"]
    designated_validation_ap = designated["validation_metrics"]["average_precision"]
    best_validation_ap = best_record["validation_metrics"]["average_precision"]

    is_competitive_on_ap = (
        designated_test_ap >= best_test_ap - 0.02
        and designated_validation_ap >= best_validation_ap - 0.02
    )
    supports_inference = (
        designated["interactive_inference_supported"]
        and designated["batch_inference_supported"]
    )
    threshold_context = designated["threshold_context"]
    has_threshold_governance = (
        threshold_context["f1_threshold"] is not None
        and threshold_context["cost_threshold"] is not None
    )
    acceptable_brier = designated["test_metrics"]["brier_score"] <= 0.06

    if is_competitive_on_ap and supports_inference and has_threshold_governance and acceptable_brier:
        return designated
    return best_record


def build_champion_decision_payload(model_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the normalized comparison table plus champion rationale."""
    ranked_records = rank_model_records(model_records)
    champion = select_champion_record(model_records)

    rationale_points = [
        (
            f"{champion['label']} ranked first on validation PR-AUC "
            f"({champion['validation_metrics']['average_precision']:.4f})"
        ),
        (
            f"{champion['label']} remained strongest on test PR-AUC "
            f"({champion['test_metrics']['average_precision']:.4f})"
        ),
        (
            f"{champion['label']} showed the lowest test Brier score "
            f"({champion['test_metrics']['brier_score']:.4f})"
        ),
    ]

    if champion["threshold_context"]["f1_threshold"] is not None:
        rationale_points.append(
            "The selected model already includes persisted threshold-selection and cost-analysis artifacts."
        )
    if champion["interactive_inference_supported"] and champion["batch_inference_supported"]:
        rationale_points.append(
            "The selected model is already wired into both interactive and batch inference paths."
        )

    comparison_table: list[dict[str, Any]] = []
    for rank, record in enumerate(ranked_records, start=1):
        comparison_table.append(
            {
                "rank": rank,
                "model_key": record["model_key"],
                "label": record["label"],
                "model_family": record["model_family"],
                "validation_average_precision": record["validation_metrics"]["average_precision"],
                "validation_roc_auc": record["validation_metrics"]["roc_auc"],
                "validation_ks_statistic": record["validation_metrics"]["ks_statistic"],
                "validation_brier_score": record["validation_metrics"]["brier_score"],
                "test_average_precision": record["test_metrics"]["average_precision"],
                "test_roc_auc": record["test_metrics"]["roc_auc"],
                "test_ks_statistic": record["test_metrics"]["ks_statistic"],
                "test_brier_score": record["test_metrics"]["brier_score"],
                "selected_candidate_source": record["selected_candidate_source"],
                "interactive_inference_supported": record["interactive_inference_supported"],
                "batch_inference_supported": record["batch_inference_supported"],
            }
        )

    return {
        "comparison_table": comparison_table,
        "champion_model_key": champion["model_key"],
        "champion_label": champion["label"],
        "champion_reasoning": rationale_points,
        "ranked_model_keys": [record["model_key"] for record in ranked_records],
    }


def default_model_comparison_configs(project_root: Path) -> list[ModelComparisonConfig]:
    """Return the default candidate set for the current project."""
    artifact_root = project_root / "artifacts"
    return [
        ModelComparisonConfig(
            model_key="baseline",
            label="Logistic Regression Baseline",
            model_family="sklearn_logistic_regression",
            artifact_dir=artifact_root / "baseline",
            metrics_file="baseline_metrics.json",
            run_summary_file="run_summary.json",
        ),
        ModelComparisonConfig(
            model_key="xgboost",
            label="XGBoost Champion Candidate",
            model_family="xgboost",
            artifact_dir=artifact_root / "xgboost",
            metrics_file="xgboost_metrics.json",
            run_summary_file="run_summary.json",
            threshold_selection_file="threshold_selection_report.json",
            cost_analysis_file="cost_analysis_report.json",
            calibration_file="calibration_report.json",
            interactive_inference_supported=True,
            batch_inference_supported=True,
            designated_champion=True,
        ),
        ModelComparisonConfig(
            model_key="torch",
            label="PyTorch MLP Challenger",
            model_family="pytorch_mlp",
            artifact_dir=artifact_root / "torch",
            metrics_file="torch_metrics.json",
            run_summary_file="run_summary.json",
        ),
    ]


def build_default_comparison_payload(project_root: Path) -> dict[str, Any]:
    """Build the current default comparison payload from persisted artifacts."""
    configs = default_model_comparison_configs(project_root)
    records = [build_model_comparison_record(config) for config in configs]
    decision_payload = build_champion_decision_payload(records)
    return {
        "generated_from": [asdict(config) for config in configs],
        "model_records": records,
        "decision": decision_payload,
    }
