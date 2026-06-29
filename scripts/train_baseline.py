#!/usr/bin/env python3
"""Train the Stage 3 logistic-regression baseline and persist its artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_risk_control.config import get_settings
from ml_risk_control.data.repositories import (
    RepositoryValidationError,
    build_repository,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the logistic-regression baseline and save reproducible artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baseline",
        help="Directory where the baseline artifact bundle will be written.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for evaluation outputs. Defaults to project settings.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="0.1.0",
        help="Semantic version to store inside the baseline artifact.",
    )
    parser.add_argument(
        "--schema-version",
        type=str,
        default="1.0.0",
        help="Feature schema version to store alongside the baseline artifact.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Optional class-weight strategy for the baseline classifier.",
    )
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2),
        encoding="utf-8",
    )


def _load_training_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from ml_risk_control.evaluation.metrics import evaluate_binary_classifier
        from ml_risk_control.features.build import (
            SplitConfig,
            build_split_metadata,
            split_training_data,
        )
        from ml_risk_control.models.baseline import (
            LogisticRegressionBaseline,
            LogisticRegressionBaselineConfig,
        )
    except ImportError as error:
        print(
            "ERROR: Missing baseline-training dependencies. "
            "Install the project requirements before running this script.",
            file=sys.stderr,
        )
        print(f"DETAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    return (
        evaluate_binary_classifier,
        SplitConfig,
        build_split_metadata,
        split_training_data,
        LogisticRegressionBaseline,
        LogisticRegressionBaselineConfig,
    )


def main() -> int:
    args = parse_args()
    settings = get_settings()
    threshold = (
        settings.training.decision_threshold
        if args.threshold is None
        else args.threshold
    )

    if not 0.0 <= threshold <= 1.0:
        print("ERROR: threshold must be bounded within [0, 1].", file=sys.stderr)
        return 1

    (
        evaluate_binary_classifier,
        SplitConfig,
        build_split_metadata,
        split_training_data,
        LogisticRegressionBaseline,
        LogisticRegressionBaselineConfig,
    ) = _load_training_modules()

    repository = build_repository(settings)
    validation_report = repository.validate_raw_data()
    validation_errors = [
        error
        for file_report in validation_report["files"].values()
        for error in file_report["errors"]
    ]
    if validation_errors:
        for error in validation_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    try:
        training_data = repository.load_training_data()
    except RepositoryValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    split_config = SplitConfig(random_state=settings.training.random_state)
    partitions = split_training_data(
        training_data,
        target_column=settings.training.target_column,
        config=split_config,
    )

    classifier_config = LogisticRegressionBaselineConfig(
        class_weight=None if args.class_weight == "none" else args.class_weight,
        random_state=settings.training.random_state,
    )
    baseline = LogisticRegressionBaseline(
        id_column=settings.training.id_column,
        target_column=settings.training.target_column,
        model_version=args.model_version,
        schema_version=args.schema_version,
        config=classifier_config,
    ).fit(partitions.train)

    split_metadata = build_split_metadata(
        partitions,
        target_column=settings.training.target_column,
        config=split_config,
    )
    split_metadata["generated_at_utc"] = datetime.now(UTC).isoformat()
    split_metadata["dataset_fingerprint"] = validation_report.get("dataset_fingerprint")

    partition_metrics: dict[str, Any] = {}
    for partition_name, partition_frame in {
        "train": partitions.train,
        "validation": partitions.validation,
        "test": partitions.test,
    }.items():
        y_true = partition_frame[settings.training.target_column]
        y_score = baseline.predict_proba(partition_frame)
        partition_metrics[partition_name] = evaluate_binary_classifier(
            y_true,
            y_score,
            threshold=threshold,
        )

    metrics_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_name": "logistic_regression_baseline",
        "model_version": args.model_version,
        "schema_version": args.schema_version,
        "backend": repository.backend,
        "dataset_fingerprint": validation_report.get("dataset_fingerprint"),
        "threshold": threshold,
        "partitions": partition_metrics,
    }

    feature_schema_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "feature_schema": baseline.feature_schema.to_dict(),
    }

    run_summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_name": "logistic_regression_baseline",
        "model_version": args.model_version,
        "schema_version": args.schema_version,
        "backend": repository.backend,
        "dataset_fingerprint": validation_report.get("dataset_fingerprint"),
        "training_summary": baseline.training_summary_,
        "artifact_metadata": baseline.build_artifact_metadata(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_artifact_path = args.output_dir / "logistic_regression_baseline.joblib"
    feature_schema_path = args.output_dir / "feature_schema.json"
    split_metadata_path = args.output_dir / "split_metadata.json"
    metrics_path = args.output_dir / "baseline_metrics.json"
    run_summary_path = args.output_dir / "run_summary.json"

    baseline.save(model_artifact_path)
    _write_json(feature_schema_path, feature_schema_payload)
    _write_json(split_metadata_path, split_metadata)
    _write_json(metrics_path, metrics_payload)
    _write_json(run_summary_path, run_summary)

    print("Baseline training completed.")
    print(f"Model artifact: {model_artifact_path}")
    print(f"Feature schema: {feature_schema_path}")
    print(f"Split metadata: {split_metadata_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Run summary: {run_summary_path}")
    print(
        "Validation PR-AUC: "
        f"{partition_metrics['validation']['average_precision']:.6f}"
    )
    print(
        "Validation ROC-AUC: "
        f"{partition_metrics['validation']['roc_auc']:.6f}"
    )
    print(
        "Validation KS: "
        f"{partition_metrics['validation']['ks_statistic']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
