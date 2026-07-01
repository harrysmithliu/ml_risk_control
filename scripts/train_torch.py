#!/usr/bin/env python3
"""Train the Stage 7 PyTorch challenger and persist its artifacts."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

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
        description="Train the PyTorch MLP challenger and save reproducible artifacts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model_torch.yaml",
        help="Path to the PyTorch challenger YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for the artifact output directory.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for the evaluation threshold.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Optional override for the saved model version.",
    )
    parser.add_argument(
        "--schema-version",
        type=str,
        default=None,
        help="Optional override for the saved feature schema version.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose PyTorch training output.",
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
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if inner == "":
            return []
        return [_parse_scalar(item) for item in inner.split(",")]
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    entries: list[tuple[int, str]] = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        entries.append((len(line) - len(line.lstrip(" ")), line.strip()))

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for index, (indent, content) in enumerate(entries):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith("- "):
            if not isinstance(parent, list):
                msg = f"Invalid YAML structure near list item: {content}"
                raise ValueError(msg)
            parent.append(_parse_scalar(content[2:]))
            continue

        if ":" not in content:
            msg = f"Unsupported YAML line: {content}"
            raise ValueError(msg)

        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        if raw_value:
            if not isinstance(parent, dict):
                msg = f"Invalid YAML mapping near: {content}"
                raise ValueError(msg)
            parent[key] = _parse_scalar(raw_value)
            continue

        if index + 1 < len(entries) and entries[index + 1][0] > indent:
            next_content = entries[index + 1][1]
            next_container: dict[str, Any] | list[Any]
            next_container = [] if next_content.startswith("- ") else {}
        else:
            next_container = {}

        if not isinstance(parent, dict):
            msg = f"Invalid YAML nesting near: {content}"
            raise ValueError(msg)
        parent[key] = next_container
        stack.append((indent, next_container))

    return root


def _load_training_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from ml_risk_control.evaluation.metrics import evaluate_binary_classifier
        from ml_risk_control.features.build import (
            SplitConfig,
            build_split_metadata,
            split_training_data,
        )
        from ml_risk_control.models.torch_model import (
            TorchMLPConfig,
            TorchMLPCreditRiskModel,
        )
    except ImportError as error:
        print(
            "ERROR: Missing PyTorch challenger dependencies. "
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
        TorchMLPCreditRiskModel,
        TorchMLPConfig,
    )


def _build_torch_config(
    config_payload: dict[str, Any],
    *,
    settings_random_state: int,
) -> dict[str, Any]:
    runtime = config_payload.get("runtime", {})
    training = config_payload.get("training", {})
    return {
        "hidden_dims": tuple(training.get("hidden_dims", [64, 32])),
        "dropout": float(training.get("dropout", 0.10)),
        "learning_rate": float(training.get("learning_rate", 1e-3)),
        "batch_size": int(training.get("batch_size", 512)),
        "max_epochs": int(training.get("max_epochs", 50)),
        "patience": int(training.get("patience", 8)),
        "min_delta": float(training.get("min_delta", 1e-4)),
        "weight_decay": float(training.get("weight_decay", 1e-4)),
        "positive_class_weight_strategy": str(
            training.get("positive_class_weight_strategy", "auto_from_train_ratio")
        ),
        "positive_class_weight_value": training.get("positive_class_weight_value"),
        "random_state": int(runtime.get("random_state", settings_random_state)),
        "device": str(runtime.get("device", "cpu")),
    }


def _assert_finite_probabilities(probabilities: pd.Series, *, partition_name: str) -> None:
    if probabilities.isna().any():
        msg = f"Reload check failed: {partition_name} probabilities contain NaN values."
        raise ValueError(msg)
    if not probabilities.map(math.isfinite).all():
        msg = f"Reload check failed: {partition_name} probabilities contain non-finite values."
        raise ValueError(msg)


def main() -> int:
    args = parse_args()
    settings = get_settings()
    config_payload = _load_simple_yaml(args.config)
    metadata = config_payload.get("metadata", {})
    artifacts = config_payload.get("artifacts", {})
    guardrails = config_payload.get("guardrails", {})

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
        TorchMLPCreditRiskModel,
        TorchMLPConfig,
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

    classifier_config = TorchMLPConfig(
        **_build_torch_config(
            config_payload,
            settings_random_state=settings.training.random_state,
        )
    )
    model_version = args.model_version or metadata.get("model_version", "0.1.0")
    schema_version = args.schema_version or metadata.get("schema_version", "1.0.0")
    output_dir = args.output_dir or PROJECT_ROOT / str(
        artifacts.get("output_dir", "artifacts/torch")
    )

    challenger = TorchMLPCreditRiskModel(
        id_column=settings.training.id_column,
        target_column=settings.training.target_column,
        model_version=model_version,
        schema_version=schema_version,
        config=classifier_config,
    ).fit(
        partitions.train,
        eval_dataframe=partitions.validation,
        verbose=args.verbose,
    )

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
        y_score = challenger.predict_proba(partition_frame)
        partition_metrics[partition_name] = evaluate_binary_classifier(
            y_true,
            y_score,
            threshold=threshold,
        )

    metrics_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_name": "torch_mlp_challenger",
        "model_version": model_version,
        "schema_version": schema_version,
        "backend": repository.backend,
        "dataset_fingerprint": validation_report.get("dataset_fingerprint"),
        "threshold": threshold,
        "partitions": partition_metrics,
    }

    feature_schema_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "feature_schema": challenger.feature_schema.to_dict(),
    }

    training_history_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_name": "torch_mlp_challenger",
        "model_version": model_version,
        "schema_version": schema_version,
        "history": challenger.training_history_,
    }

    config_snapshot_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(args.config),
        "config": config_payload,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_artifact_path = output_dir / str(
        artifacts.get("model_file", "torch_mlp_challenger.pt")
    )
    config_snapshot_path = output_dir / str(
        artifacts.get("config_snapshot_file", "torch_config_snapshot.json")
    )
    feature_schema_path = output_dir / str(
        artifacts.get("feature_schema_file", "feature_schema.json")
    )
    split_metadata_path = output_dir / str(
        artifacts.get("split_metadata_file", "split_metadata.json")
    )
    metrics_path = output_dir / str(artifacts.get("metrics_file", "torch_metrics.json"))
    run_summary_path = output_dir / str(artifacts.get("run_summary_file", "run_summary.json"))
    training_history_path = output_dir / str(
        artifacts.get("training_history_file", "training_history.json")
    )

    challenger.save(model_artifact_path)
    reloaded = TorchMLPCreditRiskModel.load(model_artifact_path)
    reload_probabilities = reloaded.predict_proba(partitions.validation)
    if guardrails.get("require_finite_probabilities", True):
        _assert_finite_probabilities(reload_probabilities, partition_name="validation")

    run_summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_name": "torch_mlp_challenger",
        "model_version": model_version,
        "schema_version": schema_version,
        "backend": repository.backend,
        "dataset_fingerprint": validation_report.get("dataset_fingerprint"),
        "config_path": str(args.config),
        "training_summary": challenger.training_summary_,
        "artifact_metadata": challenger.build_artifact_metadata(),
        "reload_check": {
            "status": "passed",
            "partition": "validation",
            "row_count": int(len(reload_probabilities)),
            "min_probability": float(reload_probabilities.min()),
            "max_probability": float(reload_probabilities.max()),
        },
    }

    _write_json(config_snapshot_path, config_snapshot_payload)
    _write_json(feature_schema_path, feature_schema_payload)
    _write_json(split_metadata_path, split_metadata)
    _write_json(metrics_path, metrics_payload)
    _write_json(training_history_path, training_history_payload)
    _write_json(run_summary_path, run_summary)

    print("PyTorch challenger training completed.")
    print(f"Model artifact: {model_artifact_path}")
    print(f"Feature schema: {feature_schema_path}")
    print(f"Split metadata: {split_metadata_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Training history: {training_history_path}")
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
