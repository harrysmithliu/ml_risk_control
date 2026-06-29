#!/usr/bin/env python3
"""Render model diagnostic PNG figures from persisted XGBoost artifacts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(tempfile.gettempdir()) / "ml_risk_control_plot_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "xgboost"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures" / "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render PR/ROC and feature-importance figures from XGBoost artifacts."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing the persisted XGBoost JSON artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG figures should be written.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="validation",
        help="Partition to visualize for PR/ROC curves. Defaults to validation.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of features to keep in importance bar charts.",
    )
    parser.add_argument(
        "--native-importance-type",
        type=str,
        default="gain",
        help="Native XGBoost importance type to plot. Defaults to gain.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_positive_int(value: int, *, name: str) -> None:
    if value <= 0:
        msg = f"{name} must be positive."
        raise ValueError(msg)


def _load_required_artifacts(artifact_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    curves_path = artifact_dir / "curves.json"
    native_importance_path = artifact_dir / "native_feature_importance.json"
    permutation_importance_path = artifact_dir / "permutation_importance.json"

    missing_paths = [
        path
        for path in (curves_path, native_importance_path, permutation_importance_path)
        if not path.exists()
    ]
    if missing_paths:
        missing_display = ", ".join(str(path) for path in missing_paths)
        msg = f"Missing required artifact files: {missing_display}"
        raise FileNotFoundError(msg)

    return (
        _read_json(curves_path),
        _read_json(native_importance_path),
        _read_json(permutation_importance_path),
    )


def _plot_precision_recall_curve(
    *,
    payload: dict[str, Any],
    partition: str,
    output_path: Path,
) -> None:
    partition_payload = payload["partitions"].get(partition)
    if partition_payload is None:
        msg = f"Partition '{partition}' was not found in curves.json."
        raise ValueError(msg)

    curve = partition_payload.get("precision_recall_curve")
    if curve is None:
        msg = f"Precision-recall curve is missing for partition '{partition}'."
        raise ValueError(msg)

    precision = curve["precision"]
    recall = curve["recall"]
    baseline_positive_rate = curve["baseline_positive_rate"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#2563eb", linewidth=2, label="PR curve")
    ax.axhline(
        baseline_positive_rate,
        color="#9ca3af",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline positive rate = {baseline_positive_rate:.4f}",
    )
    ax.set_title(f"Precision-Recall Curve ({partition.title()})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_roc_curve(
    *,
    payload: dict[str, Any],
    partition: str,
    output_path: Path,
) -> None:
    partition_payload = payload["partitions"].get(partition)
    if partition_payload is None:
        msg = f"Partition '{partition}' was not found in curves.json."
        raise ValueError(msg)

    curve = partition_payload.get("roc_curve")
    if curve is None:
        msg = f"ROC curve is missing for partition '{partition}'."
        raise ValueError(msg)

    false_positive_rate = curve["false_positive_rate"]
    true_positive_rate = curve["true_positive_rate"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(false_positive_rate, true_positive_rate, color="#dc2626", linewidth=2, label="ROC curve")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#9ca3af", linewidth=1.5, label="Random baseline")
    ax.set_title(f"ROC Curve ({partition.title()})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_native_importance(
    *,
    payload: dict[str, Any],
    importance_type: str,
    top_n: int,
    output_path: Path,
) -> None:
    native_importance = payload.get("native_importance", {})
    importance_values = native_importance.get(importance_type)
    if importance_values is None:
        available = ", ".join(sorted(native_importance.keys()))
        msg = (
            f"Native importance type '{importance_type}' was not found. "
            f"Available types: {available}"
        )
        raise ValueError(msg)

    series = (
        pd.Series(importance_values, dtype=float)
        .sort_values(ascending=False)
        .head(top_n)
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(series.index.tolist(), series.values.tolist(), color="#0f766e")
    ax.set_title(f"Native XGBoost Importance ({importance_type})")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_permutation_importance(
    *,
    payload: dict[str, Any],
    top_n: int,
    output_path: Path,
) -> None:
    feature_rows = payload.get("features", [])
    if not feature_rows:
        msg = "Permutation importance payload does not contain feature rows."
        raise ValueError(msg)

    frame = pd.DataFrame(feature_rows)
    if "feature_name" not in frame.columns or "mean_importance" not in frame.columns:
        msg = "Permutation importance payload is missing required columns."
        raise ValueError(msg)

    frame = (
        frame.loc[:, ["feature_name", "mean_importance", "std_importance"]]
        .sort_values("mean_importance", ascending=False)
        .head(top_n)
        .sort_values("mean_importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        frame["feature_name"],
        frame["mean_importance"],
        xerr=frame["std_importance"],
        color="#7c3aed",
        ecolor="#c4b5fd",
        capsize=3,
    )
    ax.set_title(
        "Permutation Importance "
        f"({payload.get('partition', 'validation').title()}, {payload.get('scoring_metric', 'metric')})"
    )
    ax.set_xlabel("Mean performance drop after permutation")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    _validate_positive_int(args.top_n, name="top_n")

    curves_payload, native_importance_payload, permutation_importance_payload = _load_required_artifacts(
        args.artifact_dir
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pr_curve_path = args.output_dir / f"pr_curve_{args.partition}.png"
    roc_curve_path = args.output_dir / f"roc_curve_{args.partition}.png"
    native_importance_path = args.output_dir / f"native_importance_{args.native_importance_type}.png"
    permutation_importance_path = args.output_dir / "permutation_importance.png"

    _plot_precision_recall_curve(
        payload=curves_payload,
        partition=args.partition,
        output_path=pr_curve_path,
    )
    _plot_roc_curve(
        payload=curves_payload,
        partition=args.partition,
        output_path=roc_curve_path,
    )
    _plot_native_importance(
        payload=native_importance_payload,
        importance_type=args.native_importance_type,
        top_n=args.top_n,
        output_path=native_importance_path,
    )
    _plot_permutation_importance(
        payload=permutation_importance_payload,
        top_n=args.top_n,
        output_path=permutation_importance_path,
    )

    print(f"Rendered: {pr_curve_path}")
    print(f"Rendered: {roc_curve_path}")
    print(f"Rendered: {native_importance_path}")
    print(f"Rendered: {permutation_importance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
