#!/usr/bin/env python3
"""Train the Stage 4 XGBoost reference candidate and persist its artifacts."""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
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
        description="Train the XGBoost reference candidate and save reproducible artifacts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model_xgb.yaml",
        help="Path to the XGBoost YAML configuration file.",
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
        help="Enable verbose XGBoost training output.",
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

        next_container: dict[str, Any] | list[Any]
        if index + 1 < len(entries) and entries[index + 1][0] > indent:
            next_content = entries[index + 1][1]
            next_container = [] if next_content.startswith("- ") else {}
        else:
            next_container = {}

        if not isinstance(parent, dict):
            msg = f"Invalid YAML nesting near: {content}"
            raise ValueError(msg)
        parent[key] = next_container
        stack.append((indent, next_container))

    return root


def _load_training_modules() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from ml_risk_control.evaluation.metrics import evaluate_binary_classifier
        from ml_risk_control.features.build import (
            SplitConfig,
            build_split_metadata,
            split_training_data,
        )
        from ml_risk_control.models.xgboost_model import (
            XGBoostCreditRiskModel,
            XGBoostModelConfig,
        )
    except ImportError as error:
        print(
            "ERROR: Missing XGBoost-training dependencies. "
            "Install project requirements and platform runtime dependencies before running this script.",
            file=sys.stderr,
        )
        print(f"DETAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    return (
        evaluate_binary_classifier,
        SplitConfig,
        build_split_metadata,
        split_training_data,
        (XGBoostCreditRiskModel, XGBoostModelConfig),
    )


def _build_xgb_config_payload(config_payload: dict[str, Any], *, settings_random_state: int) -> dict[str, Any]:
    runtime = config_payload.get("runtime", {})
    training = config_payload.get("training", {})
    reference_run = config_payload.get("reference_run", {})
    params = reference_run.get("params", {})

    return {
        "objective": training.get("objective", "binary:logistic"),
        "eval_metric": tuple(training.get("eval_metric", ["aucpr", "auc", "logloss"])),
        "early_stopping_rounds": training.get("early_stopping_rounds", 50),
        "tree_method": runtime.get("tree_method", "hist"),
        "device": runtime.get("device", "cpu"),
        "n_jobs": runtime.get("n_jobs", -1),
        "random_state": runtime.get("random_state", settings_random_state),
        "verbosity": runtime.get("verbosity", 1),
        **params,
    }


def _get_metric_value(metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name not in metrics:
        msg = f"Selection metric '{metric_name}' is missing from evaluation metrics."
        raise ValueError(msg)
    return float(metrics[metric_name])


def _is_better_score(candidate_score: float, incumbent_score: float, *, direction: str) -> bool:
    if direction == "minimize":
        return candidate_score < incumbent_score
    return candidate_score > incumbent_score


def _summarize_metrics(
    metrics: dict[str, Any],
    *,
    metric_names: list[str],
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name in metric_names:
        if metric_name in metrics:
            summary[metric_name] = float(metrics[metric_name])
    return summary


def _build_candidate_summary(
    candidate_result: dict[str, Any],
    *,
    metric_names: list[str],
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "candidate_name": candidate_result["candidate_name"],
        "params": candidate_result["params"],
        "training_summary": candidate_result["training_summary"],
        "validation_metric_summary": _summarize_metrics(
            candidate_result["validation_metrics"],
            metric_names=metric_names,
        ),
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload


def _sample_tuning_parameter_sets(
    tuning_payload: dict[str, Any],
    *,
    random_state: int,
) -> list[dict[str, Any]]:
    parameter_space = tuning_payload.get("parameter_space", {})
    if not parameter_space:
        return []

    normalized_space: dict[str, list[Any]] = {}
    for parameter_name, parameter_payload in parameter_space.items():
        if isinstance(parameter_payload, dict):
            values = parameter_payload.get("values", [])
        else:
            values = parameter_payload
        if not values:
            msg = f"Tuning parameter '{parameter_name}' has no candidate values."
            raise ValueError(msg)
        normalized_space[parameter_name] = list(values)

    requested_iterations = int(tuning_payload.get("n_iter", 0))
    if requested_iterations <= 0:
        return []

    total_combinations = 1
    for values in normalized_space.values():
        total_combinations *= len(values)

    target_count = min(requested_iterations, total_combinations)
    generator = random.Random(random_state)
    sampled_signatures: set[tuple[tuple[str, Any], ...]] = set()
    sampled_parameter_sets: list[dict[str, Any]] = []
    parameter_names = list(normalized_space.keys())
    max_attempts = max(target_count * 50, 100)

    while len(sampled_parameter_sets) < target_count and max_attempts > 0:
        candidate = {
            parameter_name: generator.choice(normalized_space[parameter_name])
            for parameter_name in parameter_names
        }
        signature = tuple((name, candidate[name]) for name in parameter_names)
        if signature not in sampled_signatures:
            sampled_signatures.add(signature)
            sampled_parameter_sets.append(candidate)
        max_attempts -= 1

    return sampled_parameter_sets


def _resolve_scale_pos_weight_value(
    *,
    train_frame: Any,
    target_column: str,
    class_imbalance_payload: dict[str, Any],
) -> dict[str, Any]:
    strategy = str(
        class_imbalance_payload.get(
            "scale_pos_weight_strategy",
            "auto_from_train_ratio",
        )
    )
    target_series = train_frame[target_column]
    positive_count = int((target_series == 1).sum())
    negative_count = int((target_series == 0).sum())

    if positive_count <= 0:
        msg = "scale_pos_weight experiment requires at least one positive training example."
        raise ValueError(msg)
    if negative_count <= 0:
        msg = "scale_pos_weight experiment requires at least one negative training example."
        raise ValueError(msg)

    if strategy == "auto_from_train_ratio":
        value = negative_count / positive_count
    elif strategy == "manual":
        raw_value = class_imbalance_payload.get("scale_pos_weight_value")
        if raw_value is None:
            msg = "scale_pos_weight_value is required when strategy is manual."
            raise ValueError(msg)
        value = float(raw_value)
    else:
        msg = f"Unsupported scale_pos_weight strategy: {strategy}"
        raise ValueError(msg)

    if value <= 0.0:
        msg = "scale_pos_weight must be strictly positive."
        raise ValueError(msg)

    return {
        "strategy": strategy,
        "value": float(value),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": positive_count / (positive_count + negative_count),
    }


def _compute_permutation_importance(
    *,
    model: Any,
    evaluation_frame: pd.DataFrame,
    target_column: str,
    evaluate_binary_classifier: Any,
    scoring_metric: str,
    score_direction: str,
    threshold: float,
    n_repeats: int,
    random_state: int,
) -> dict[str, Any]:
    if n_repeats <= 0:
        msg = "Permutation importance requires n_repeats to be positive."
        raise ValueError(msg)

    baseline_scores = model.predict_proba(evaluation_frame)
    baseline_metrics = evaluate_binary_classifier(
        evaluation_frame[target_column],
        baseline_scores,
        threshold=threshold,
    )
    baseline_metric_value = _get_metric_value(baseline_metrics, scoring_metric)
    feature_names = getattr(model.feature_schema, "raw_feature_columns", None)
    if feature_names is None and hasattr(model.feature_schema, "to_dict"):
        feature_names = model.feature_schema.to_dict().get("raw_feature_columns")
    if not feature_names:
        msg = "Model feature schema does not expose raw feature columns for permutation importance."
        raise ValueError(msg)
    feature_names = list(feature_names)

    generator = random.Random(random_state)
    feature_importance_rows: list[dict[str, Any]] = []
    for feature_name in feature_names:
        permuted_metric_values: list[float] = []
        importance_values: list[float] = []
        for repeat_index in range(n_repeats):
            permuted_frame = evaluation_frame.copy()
            shuffle_seed = generator.randint(0, 10**9)
            permuted_values = (
                permuted_frame[feature_name]
                .sample(frac=1.0, random_state=shuffle_seed)
                .to_numpy()
            )
            permuted_frame.loc[:, feature_name] = permuted_values
            permuted_scores = model.predict_proba(permuted_frame)
            permuted_metrics = evaluate_binary_classifier(
                permuted_frame[target_column],
                permuted_scores,
                threshold=threshold,
            )
            permuted_metric_value = _get_metric_value(permuted_metrics, scoring_metric)
            if score_direction == "minimize":
                importance_value = permuted_metric_value - baseline_metric_value
            else:
                importance_value = baseline_metric_value - permuted_metric_value
            permuted_metric_values.append(permuted_metric_value)
            importance_values.append(float(importance_value))

        mean_importance = sum(importance_values) / len(importance_values)
        mean_permuted_metric = sum(permuted_metric_values) / len(permuted_metric_values)
        variance = sum(
            (value - mean_importance) ** 2 for value in importance_values
        ) / len(importance_values)
        feature_importance_rows.append(
            {
                "feature_name": feature_name,
                "mean_importance": float(mean_importance),
                "std_importance": float(variance**0.5),
                "mean_permuted_metric": float(mean_permuted_metric),
                "permuted_metric_values": [float(value) for value in permuted_metric_values],
                "importance_values": [float(value) for value in importance_values],
            }
        )

    feature_importance_rows.sort(
        key=lambda row: row["mean_importance"],
        reverse=True,
    )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "partition": "validation",
        "scoring_metric": scoring_metric,
        "score_direction": score_direction,
        "n_repeats": int(n_repeats),
        "baseline_metric_value": float(baseline_metric_value),
        "baseline_positive_rate": float((evaluation_frame[target_column] == 1).mean()),
        "features": feature_importance_rows,
    }


def _train_candidate(
    *,
    candidate_name: str,
    candidate_params: dict[str, Any],
    base_config_payload: dict[str, Any],
    model_class: Any,
    model_config_class: Any,
    settings: Any,
    model_version: str,
    schema_version: str,
    native_importance_types: tuple[str, ...],
    train_frame: Any,
    validation_frame: Any,
    evaluate_binary_classifier: Any,
    threshold: float,
    verbose: bool,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    effective_config = {**base_config_payload, **candidate_params}
    model_config = model_config_class(**effective_config)
    if hasattr(model_config, "to_dict"):
        effective_config_payload = model_config.to_dict()
    else:
        effective_config_payload = getattr(model_config, "kwargs", effective_config)
    model = model_class(
        id_column=settings.training.id_column,
        target_column=settings.training.target_column,
        model_version=str(model_version),
        schema_version=str(schema_version),
        config=model_config,
        native_importance_types=native_importance_types,
    )
    model.fit(
        train_frame,
        eval_dataframe=validation_frame,
        verbose=verbose,
    )
    validation_scores = model.predict_proba(validation_frame)
    validation_metrics = evaluate_binary_classifier(
        validation_frame[settings.training.target_column],
        validation_scores,
        threshold=threshold,
    )
    candidate_result = {
        "candidate_name": candidate_name,
        "params": candidate_params,
        "effective_config": effective_config_payload,
        "training_summary": model.training_summary_,
        "validation_metrics": validation_metrics,
    }
    return model, validation_metrics, candidate_result


def _evaluate_selected_model(
    *,
    model: Any,
    partitions: Any,
    evaluation_rows: list[str],
    settings: Any,
    evaluate_binary_classifier: Any,
    threshold: float,
) -> dict[str, Any]:
    partition_frames = {
        "train": partitions.train,
        "validation": partitions.validation,
        "test": partitions.test,
    }
    partition_metrics: dict[str, Any] = {}
    for partition_name in evaluation_rows:
        partition_frame = partition_frames[partition_name]
        y_true = partition_frame[settings.training.target_column]
        y_score = model.predict_proba(partition_frame)
        partition_metrics[partition_name] = evaluate_binary_classifier(
            y_true,
            y_score,
            threshold=threshold,
        )
    return partition_metrics


def _build_curves_payload(
    *,
    partition_metrics: dict[str, Any],
    metadata: dict[str, Any],
    model_version: str,
    schema_version: str,
    selected_candidate_source: str | None,
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_name": metadata.get("model_name", "xgboost_credit_risk"),
        "model_version": model_version,
        "schema_version": schema_version,
        "selected_candidate_source": selected_candidate_source,
        "partitions": {
            partition_name: {
                "precision_recall_curve": metrics.get("precision_recall_curve"),
                "roc_curve": metrics.get("roc_curve"),
            }
            for partition_name, metrics in partition_metrics.items()
        },
    }


def _run_reload_check(
    model_class: Any,
    artifact_path: Path,
    validation_frame: pd.DataFrame,
) -> dict[str, Any]:
    reloaded_model = model_class.load(artifact_path)
    reloaded_probabilities = reloaded_model.predict_proba(validation_frame)

    if len(reloaded_probabilities) != len(validation_frame):
        msg = "Reloaded XGBoost artifact returned an unexpected number of probabilities."
        raise ValueError(msg)

    finite_mask = reloaded_probabilities.astype(float).map(math.isfinite)
    if not bool(finite_mask.all()):
        msg = "Reloaded XGBoost artifact produced non-finite probabilities."
        raise ValueError(msg)

    min_probability = None
    max_probability = None
    if len(reloaded_probabilities) > 0:
        min_probability = float(reloaded_probabilities.min())
        max_probability = float(reloaded_probabilities.max())

    return {
        "status": "passed",
        "row_count": int(len(reloaded_probabilities)),
        "min_probability": min_probability,
        "max_probability": max_probability,
    }


def main() -> int:
    args = parse_args()
    settings = get_settings()

    if not args.config.exists():
        print(f"ERROR: Missing XGBoost config file: {args.config}", file=sys.stderr)
        return 1

    try:
        config_payload = _load_simple_yaml(args.config)
    except ValueError as error:
        print(f"ERROR: Failed to parse XGBoost config: {error}", file=sys.stderr)
        return 1

    metadata = config_payload.get("metadata", {})
    artifacts = config_payload.get("artifacts", {})
    training_section = config_payload.get("training", {})
    diagnostics = config_payload.get("diagnostics", {})
    guardrails = config_payload.get("guardrails", {})
    tuning_section = config_payload.get("tuning", {})
    reference_run = config_payload.get("reference_run", {})
    experiments_section = config_payload.get("experiments", {})
    class_imbalance_section = experiments_section.get("class_imbalance", {})

    threshold = (
        settings.training.decision_threshold
        if args.threshold is None
        else args.threshold
    )
    if "threshold" in training_section and args.threshold is None:
        threshold = float(training_section["threshold"])

    if not 0.0 <= threshold <= 1.0:
        print("ERROR: threshold must be bounded within [0, 1].", file=sys.stderr)
        return 1

    (
        evaluate_binary_classifier,
        SplitConfig,
        build_split_metadata,
        split_training_data,
        xgb_classes,
    ) = _load_training_modules()
    XGBoostCreditRiskModel, XGBoostModelConfig = xgb_classes

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

    split_random_state = int(
        config_payload.get("runtime", {}).get("random_state", settings.training.random_state)
    )
    split_config = SplitConfig(random_state=split_random_state)
    partitions = split_training_data(
        training_data,
        target_column=settings.training.target_column,
        config=split_config,
    )

    base_config_payload = _build_xgb_config_payload(
        config_payload,
        settings_random_state=settings.training.random_state,
    )
    output_dir = args.output_dir or PROJECT_ROOT / str(artifacts.get("output_dir", "artifacts/xgboost"))
    model_version = args.model_version or metadata.get("model_version", "0.1.0")
    schema_version = args.schema_version or metadata.get("schema_version", "1.0.0")
    native_importance_types = tuple(
        diagnostics.get("native_importance_types", ["gain", "weight", "cover"])
    )
    selection_metric = str(
        tuning_section.get(
            "scoring",
            {},
        ).get(
            "primary",
            training_section.get("primary_selection_metric", "average_precision"),
        )
    )
    score_direction = str(tuning_section.get("score_direction", "maximize")).lower()
    secondary_metrics = [
        str(metric_name)
        for metric_name in tuning_section.get("scoring", {}).get("secondary", [])
    ]
    comparison_metric_names = [selection_metric, *secondary_metrics]

    reference_enabled = bool(reference_run.get("enabled", True)) and bool(
        class_imbalance_section.get("run_original_distribution", True)
    )
    scale_pos_weight_enabled = bool(
        class_imbalance_section.get("run_scale_pos_weight_variant", False)
    )
    tuning_enabled = bool(tuning_section.get("enabled", False))
    if not reference_enabled and not tuning_enabled and not scale_pos_weight_enabled:
        print(
            "ERROR: Reference, scale_pos_weight experiment, and tuning are all disabled; no XGBoost candidate can be trained.",
            file=sys.stderr,
        )
        return 1

    selected_model = None
    selected_candidate_source = None
    selected_validation_score = None
    reference_candidate_result = None
    reference_validation_metrics = None
    reference_validation_score = None

    if reference_enabled:
        try:
            (
                reference_model,
                reference_validation_metrics,
                reference_candidate_result,
            ) = _train_candidate(
                candidate_name="reference",
                candidate_params=dict(base_config_payload),
                base_config_payload={},
                model_class=XGBoostCreditRiskModel,
                model_config_class=XGBoostModelConfig,
                settings=settings,
                model_version=str(model_version),
                schema_version=str(schema_version),
                native_importance_types=native_importance_types,
                train_frame=partitions.train,
                validation_frame=partitions.validation,
                evaluate_binary_classifier=evaluate_binary_classifier,
                threshold=threshold,
                verbose=args.verbose,
            )
        except ImportError as error:
            print("ERROR: XGBoost reference training could not start.", file=sys.stderr)
            print(f"DETAIL: {error}", file=sys.stderr)
            return 1

        reference_validation_score = _get_metric_value(
            reference_validation_metrics,
            selection_metric,
        )
        selected_model = reference_model
        selected_candidate_source = "reference"
        selected_validation_score = reference_validation_score

    tuning_results_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "not_run",
        "selection_metric": selection_metric,
        "score_direction": score_direction,
        "reference_candidate": None,
        "selected_candidate_source": selected_candidate_source,
        "selected_candidate": None,
        "class_imbalance_experiments": {
            "status": "not_run",
            "selected_candidate_source": selected_candidate_source,
            "scale_pos_weight_variant": None,
        },
        "search": None,
    }
    if reference_candidate_result is not None:
        tuning_results_payload["reference_candidate"] = _build_candidate_summary(
            reference_candidate_result,
            metric_names=comparison_metric_names,
        )

    if scale_pos_weight_enabled:
        try:
            scale_pos_weight_details = _resolve_scale_pos_weight_value(
                train_frame=partitions.train,
                target_column=settings.training.target_column,
                class_imbalance_payload=class_imbalance_section,
            )
            (
                scale_pos_weight_model,
                scale_pos_weight_validation_metrics,
                scale_pos_weight_candidate_result,
            ) = _train_candidate(
                candidate_name="scale_pos_weight_variant",
                candidate_params={
                    "scale_pos_weight": scale_pos_weight_details["value"],
                },
                base_config_payload=base_config_payload,
                model_class=XGBoostCreditRiskModel,
                model_config_class=XGBoostModelConfig,
                settings=settings,
                model_version=str(model_version),
                schema_version=str(schema_version),
                native_importance_types=native_importance_types,
                train_frame=partitions.train,
                validation_frame=partitions.validation,
                evaluate_binary_classifier=evaluate_binary_classifier,
                threshold=threshold,
                verbose=args.verbose,
            )
        except (ImportError, ValueError) as error:
            print(
                "ERROR: scale_pos_weight experiment could not be completed.",
                file=sys.stderr,
            )
            print(f"DETAIL: {error}", file=sys.stderr)
            return 1

        scale_pos_weight_score = _get_metric_value(
            scale_pos_weight_validation_metrics,
            selection_metric,
        )
        if (
            selected_validation_score is None
            or _is_better_score(
                scale_pos_weight_score,
                selected_validation_score,
                direction=score_direction,
            )
        ):
            selected_model = scale_pos_weight_model
            selected_candidate_source = "scale_pos_weight_variant"
            selected_validation_score = scale_pos_weight_score

        tuning_results_payload["class_imbalance_experiments"] = {
            "status": "completed",
            "selected_candidate_source": selected_candidate_source,
            "scale_pos_weight_variant": _build_candidate_summary(
                scale_pos_weight_candidate_result,
                metric_names=comparison_metric_names,
                extra_fields={
                    "strategy": scale_pos_weight_details["strategy"],
                    "computed_scale_pos_weight": scale_pos_weight_details["value"],
                    "train_distribution": {
                        "positive_count": scale_pos_weight_details["positive_count"],
                        "negative_count": scale_pos_weight_details["negative_count"],
                        "positive_rate": scale_pos_weight_details["positive_rate"],
                    },
                },
            ),
        }

    if tuning_enabled:
        if guardrails.get("allow_final_test_set_for_search", False):
            print(
                "ERROR: Search must not use the final test set during Stage 4 tuning.",
                file=sys.stderr,
            )
            return 1

        try:
            sampled_parameter_sets = _sample_tuning_parameter_sets(
                tuning_section,
                random_state=split_random_state,
            )
        except ValueError as error:
            print(f"ERROR: Invalid tuning configuration: {error}", file=sys.stderr)
            return 1

        if not sampled_parameter_sets:
            tuning_results_payload["status"] = "skipped"
            tuning_results_payload["search"] = {
                "strategy": tuning_section.get("strategy", "randomized_search"),
                "requested_iterations": int(tuning_section.get("n_iter", 0)),
                "evaluated_candidates": 0,
                "successful_candidates": 0,
                "leaderboard": [],
                "failures": [],
                "reason": "no_parameter_sets_generated",
            }
        else:
            best_tuned_model = None
            best_tuned_result = None
            best_tuned_score = None
            leaderboard: list[dict[str, Any]] = []
            failures: list[dict[str, str]] = []

            for candidate_index, candidate_params in enumerate(sampled_parameter_sets, start=1):
                candidate_name = f"tuning_candidate_{candidate_index:02d}"
                try:
                    tuned_model, tuned_validation_metrics, tuned_candidate_result = _train_candidate(
                        candidate_name=candidate_name,
                        candidate_params=candidate_params,
                        base_config_payload=base_config_payload,
                        model_class=XGBoostCreditRiskModel,
                        model_config_class=XGBoostModelConfig,
                        settings=settings,
                        model_version=str(model_version),
                        schema_version=str(schema_version),
                        native_importance_types=native_importance_types,
                        train_frame=partitions.train,
                        validation_frame=partitions.validation,
                        evaluate_binary_classifier=evaluate_binary_classifier,
                        threshold=threshold,
                        verbose=args.verbose,
                    )
                except Exception as error:
                    failures.append(
                        {
                            "candidate_name": candidate_name,
                            "error": str(error),
                        }
                    )
                    continue

                candidate_score = _get_metric_value(tuned_validation_metrics, selection_metric)
                leaderboard.append(
                    {
                        "candidate_name": candidate_name,
                        "params": candidate_params,
                        "training_summary": tuned_candidate_result["training_summary"],
                        "validation_metric_summary": _summarize_metrics(
                            tuned_validation_metrics,
                            metric_names=comparison_metric_names,
                        ),
                    }
                )

                if (
                    best_tuned_score is None
                    or _is_better_score(
                        candidate_score,
                        best_tuned_score,
                        direction=score_direction,
                    )
                ):
                    best_tuned_model = tuned_model
                    best_tuned_result = tuned_candidate_result
                    best_tuned_score = candidate_score

            leaderboard.sort(
                key=lambda item: item["validation_metric_summary"].get(selection_metric, float("-inf")),
                reverse=score_direction != "minimize",
            )

            tuning_results_payload["search"] = {
                "strategy": tuning_section.get("strategy", "randomized_search"),
                "requested_iterations": int(tuning_section.get("n_iter", 0)),
                "evaluated_candidates": len(sampled_parameter_sets),
                "successful_candidates": len(leaderboard),
                "leaderboard": leaderboard,
                "failures": failures,
            }

            if best_tuned_model is None or best_tuned_result is None or best_tuned_score is None:
                tuning_results_payload["status"] = "failed"
            else:
                tuning_results_payload["status"] = "completed"
                tuning_results_payload["selected_candidate"] = _build_candidate_summary(
                    best_tuned_result,
                    metric_names=comparison_metric_names,
                )

                if (
                    selected_model is None
                    or selected_validation_score is None
                    or _is_better_score(
                        best_tuned_score,
                        selected_validation_score,
                        direction=score_direction,
                    )
                ):
                    selected_model = best_tuned_model
                    selected_candidate_source = "tuned_search"
                    selected_validation_score = best_tuned_score

    if selected_model is None:
        print("ERROR: No XGBoost candidate was trained successfully.", file=sys.stderr)
        return 1

    tuning_results_payload["selected_candidate_source"] = selected_candidate_source
    tuning_results_payload["class_imbalance_experiments"]["selected_candidate_source"] = (
        selected_candidate_source
    )

    split_metadata = build_split_metadata(
        partitions,
        target_column=settings.training.target_column,
        config=split_config,
    )
    split_metadata["generated_at_utc"] = datetime.now(UTC).isoformat()
    split_metadata["dataset_fingerprint"] = validation_report.get("dataset_fingerprint")

    evaluation_rows = artifacts.get("evaluation_rows", ["train", "validation", "test"])
    partition_metrics = _evaluate_selected_model(
        model=selected_model,
        partitions=partitions,
        evaluation_rows=evaluation_rows,
        settings=settings,
        evaluate_binary_classifier=evaluate_binary_classifier,
        threshold=threshold,
    )

    feature_schema_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "feature_schema": selected_model.feature_schema.to_dict(),
    }
    metrics_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_name": metadata.get("model_name", "xgboost_credit_risk"),
        "model_version": model_version,
        "schema_version": schema_version,
        "backend": repository.backend,
        "dataset_fingerprint": validation_report.get("dataset_fingerprint"),
        "threshold": threshold,
        "selected_candidate_source": selected_candidate_source,
        "partitions": partition_metrics,
    }
    learning_curve_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_history": selected_model.evaluation_history_,
    }
    native_importance_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "native_importance": selected_model.export_native_importance(),
    }
    permutation_importance_payload = None
    if diagnostics.get("save_permutation_importance", True):
        permutation_importance_metric = str(
            diagnostics.get("permutation_importance_scoring", selection_metric)
        )
        permutation_importance_repeats = int(
            diagnostics.get("permutation_importance_repeats", 5)
        )
        try:
            permutation_importance_payload = _compute_permutation_importance(
                model=selected_model,
                evaluation_frame=partitions.validation,
                target_column=settings.training.target_column,
                evaluate_binary_classifier=evaluate_binary_classifier,
                scoring_metric=permutation_importance_metric,
                score_direction=score_direction,
                threshold=threshold,
                n_repeats=permutation_importance_repeats,
                random_state=split_random_state,
            )
        except ValueError as error:
            print(
                "ERROR: Permutation importance computation failed.",
                file=sys.stderr,
            )
            print(f"DETAIL: {error}", file=sys.stderr)
            return 1
    curves_payload = _build_curves_payload(
        partition_metrics=partition_metrics,
        metadata=metadata,
        model_version=str(model_version),
        schema_version=str(schema_version),
        selected_candidate_source=selected_candidate_source,
    )
    config_snapshot_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "config": config_payload,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_artifact_path = output_dir / str(artifacts.get("model_file", "xgboost_credit_risk.joblib"))
    config_snapshot_path = output_dir / str(
        artifacts.get("config_snapshot_file", "xgboost_config_snapshot.json")
    )
    feature_schema_path = output_dir / str(artifacts.get("feature_schema_file", "feature_schema.json"))
    split_metadata_path = output_dir / str(artifacts.get("split_metadata_file", "split_metadata.json"))
    metrics_path = output_dir / str(artifacts.get("metrics_file", "xgboost_metrics.json"))
    tuning_results_path = output_dir / str(artifacts.get("tuning_results_file", "tuning_results.json"))
    run_summary_path = output_dir / str(artifacts.get("run_summary_file", "run_summary.json"))
    learning_curve_path = output_dir / str(
        diagnostics.get("learning_curve_file", "learning_curve.json")
    )
    curves_path = output_dir / str(diagnostics.get("curves_file", "curves.json"))
    native_importance_path = output_dir / str(
        diagnostics.get("feature_importance_file", "native_feature_importance.json")
    )
    permutation_importance_path = output_dir / str(
        diagnostics.get("permutation_importance_file", "permutation_importance.json")
    )

    selected_model.save(model_artifact_path)

    reload_check_result = None
    if guardrails.get("require_reload_check", True):
        try:
            reload_check_result = _run_reload_check(
                XGBoostCreditRiskModel,
                model_artifact_path,
                partitions.validation,
            )
        except Exception as error:
            print("ERROR: XGBoost artifact reload check failed.", file=sys.stderr)
            print(f"DETAIL: {error}", file=sys.stderr)
            return 1

    run_summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_name": metadata.get("model_name", "xgboost_credit_risk"),
        "model_version": model_version,
        "schema_version": schema_version,
        "backend": repository.backend,
        "dataset_fingerprint": validation_report.get("dataset_fingerprint"),
        "config_path": str(args.config),
        "selected_candidate_source": selected_candidate_source,
        "training_summary": selected_model.training_summary_,
        "artifact_metadata": selected_model.build_artifact_metadata(),
        "reload_check": reload_check_result,
    }

    _write_json(config_snapshot_path, config_snapshot_payload)
    _write_json(feature_schema_path, feature_schema_payload)
    _write_json(split_metadata_path, split_metadata)
    _write_json(metrics_path, metrics_payload)
    _write_json(tuning_results_path, tuning_results_payload)
    _write_json(run_summary_path, run_summary)
    if diagnostics.get("save_learning_curve", True):
        _write_json(learning_curve_path, learning_curve_payload)
    if diagnostics.get("save_curves", True):
        _write_json(curves_path, curves_payload)
    if diagnostics.get("save_feature_importance_report", True):
        _write_json(native_importance_path, native_importance_payload)
    if diagnostics.get("save_permutation_importance", True) and permutation_importance_payload is not None:
        _write_json(permutation_importance_path, permutation_importance_payload)

    print("XGBoost training completed.")
    print(f"Model artifact: {model_artifact_path}")
    print(f"Feature schema: {feature_schema_path}")
    print(f"Split metadata: {split_metadata_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Run summary: {run_summary_path}")
    if diagnostics.get("save_curves", True):
        print(f"Curves: {curves_path}")
    if diagnostics.get("save_permutation_importance", True):
        print(f"Permutation importance: {permutation_importance_path}")
    print(f"Selected candidate source: {selected_candidate_source}")
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
