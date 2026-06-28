#!/usr/bin/env python3
"""Run a reproducible exploratory data analysis pass on the raw training dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MATPLOTLIB_CONFIG_DIR = PROJECT_ROOT / "tmp" / "matplotlib"
FONTCONFIG_CACHE_DIR = PROJECT_ROOT / "tmp" / "fontconfig"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
FONTCONFIG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "tmp"))
os.environ.setdefault("FONTCONFIG_PATH", "/opt/homebrew/etc/fonts")
os.environ.setdefault("FONTCONFIG_FILE", "/opt/homebrew/etc/fonts/fonts.conf")

import pandas as pd
try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - depends on optional environment setup
    matplotlib = None
    plt = None

from ml_risk_control.config import get_settings
from ml_risk_control.data.repositories import build_repository

SELECTED_DISTRIBUTION_COLUMNS = [
    "age",
    "MonthlyIncome",
    "DebtRatio",
    "RevolvingUtilizationOfUnsecuredLines",
]

DELINQUENCY_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an initial EDA summary and generate basic figures."
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "eda" / "eda_summary.json",
        help="Path to the generated JSON summary.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "eda",
        help="Directory for generated EDA figures.",
    )
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def build_target_summary(dataframe: pd.DataFrame, target_column: str) -> dict[str, Any]:
    counts = dataframe[target_column].value_counts(dropna=False).sort_index()
    total = int(len(dataframe))
    positive_count = int(counts.get(1, 0))
    negative_count = int(counts.get(0, 0))
    return {
        "counts": {str(index): int(value) for index, value in counts.items()},
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": positive_count / total if total else None,
    }


def build_missingness_summary(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    total_rows = len(dataframe)
    for column, missing_count in dataframe.isna().sum().sort_values(ascending=False).items():
        if int(missing_count) == 0:
            continue
        summary.append(
            {
                "column": column,
                "missing_count": int(missing_count),
                "missing_rate": float(missing_count / total_rows) if total_rows else None,
            }
        )
    return summary


def build_numeric_summary(dataframe: pd.DataFrame) -> dict[str, dict[str, Any]]:
    numeric = dataframe.select_dtypes(include=["number"])
    summary = numeric.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).transpose()
    summary = summary.round(6)
    return {
        column: {metric: _json_ready(value) for metric, value in values.items()}
        for column, values in summary.to_dict(orient="index").items()
    }


def build_extreme_value_summary(
    dataframe: pd.DataFrame,
    *,
    excluded_columns: set[str],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
    for column in numeric_columns:
        if column in excluded_columns:
            continue
        series = dataframe[column].dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        summary.append(
            {
                "column": column,
                "min": float(series.min()),
                "p99": float(series.quantile(0.99)),
                "max": float(series.max()),
                "iqr_outlier_count": int(outlier_mask.sum()),
                "iqr_outlier_rate": float(outlier_mask.mean()),
            }
        )
    return sorted(summary, key=lambda item: item["iqr_outlier_count"], reverse=True)


def build_suspicious_value_summary(dataframe: pd.DataFrame) -> dict[str, Any]:
    summary = {
        "age_non_positive_count": int((dataframe["age"] <= 0).sum()),
        "age_below_18_count": int((dataframe["age"] < 18).sum()),
        "age_above_100_count": int((dataframe["age"] > 100).sum()),
        "revolving_utilization_above_1_count": int(
            (dataframe["RevolvingUtilizationOfUnsecuredLines"] > 1).sum()
        ),
        "debt_ratio_above_5_count": int((dataframe["DebtRatio"] > 5).sum()),
        "monthly_income_non_positive_count": int(
            dataframe["MonthlyIncome"].fillna(0).le(0).sum()
        ),
        "dependents_negative_count": int(
            dataframe["NumberOfDependents"].fillna(0).lt(0).sum()
        ),
        "delinquency_count_above_20": {
            column: int((dataframe[column] > 20).sum()) for column in DELINQUENCY_COLUMNS
        },
        "delinquency_count_above_50": {
            column: int((dataframe[column] > 50).sum()) for column in DELINQUENCY_COLUMNS
        },
    }
    return summary


def plot_target_balance(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    output_path: Path,
) -> None:
    if plt is None:
        return
    counts = dataframe[target_column].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["#4C78A8", "#E45756"])
    ax.set_title("Target Class Balance")
    ax.set_xlabel(target_column)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_missingness(dataframe: pd.DataFrame, *, output_path: Path) -> None:
    if plt is None:
        return
    missing = dataframe.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values detected", ha="center", va="center")
        ax.set_axis_off()
    else:
        missing.plot(kind="bar", ax=ax, color="#72B7B2")
        ax.set_title("Missing Value Counts")
        ax.set_xlabel("Column")
        ax.set_ylabel("Missing Count")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_selected_distributions(dataframe: pd.DataFrame, *, output_path: Path) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for axis, column in zip(axes, SELECTED_DISTRIBUTION_COLUMNS, strict=False):
        series = dataframe[column].dropna()
        if series.empty:
            axis.text(0.5, 0.5, f"No data for {column}", ha="center", va="center")
            axis.set_axis_off()
            continue
        clipped = series.clip(upper=series.quantile(0.99))
        axis.hist(clipped, bins=40, color="#54A24B", alpha=0.85)
        axis.set_title(f"{column} (clipped at p99)")
        axis.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_delinquency_counts(dataframe: pd.DataFrame, *, output_path: Path) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, column in zip(axes, DELINQUENCY_COLUMNS, strict=False):
        clipped = dataframe[column].clip(upper=dataframe[column].quantile(0.99))
        axis.hist(clipped, bins=30, color="#F58518", alpha=0.85)
        axis.set_title(f"{column} (clipped at p99)")
        axis.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_eda_summary(dataframe: pd.DataFrame, *, target_column: str, id_column: str) -> dict[str, Any]:
    return {
        "row_count": int(len(dataframe)),
        "column_count": int(dataframe.shape[1]),
        "duplicate_row_count": int(dataframe.duplicated().sum()),
        "target_summary": build_target_summary(dataframe, target_column),
        "missingness_summary": build_missingness_summary(dataframe),
        "numeric_summary": build_numeric_summary(dataframe),
        "extreme_value_summary": build_extreme_value_summary(
            dataframe,
            excluded_columns={id_column, target_column},
        ),
        "suspicious_value_summary": build_suspicious_value_summary(dataframe),
    }


def main() -> int:
    args = parse_args()
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    repository = build_repository(settings)
    training_data = repository.load_training_data()

    summary = build_eda_summary(
        training_data,
        target_column=settings.training.target_column,
        id_column=settings.training.id_column,
    )
    summary["generated_at_utc"] = datetime.now(UTC).isoformat()
    summary["backend"] = repository.backend
    summary["source"] = str(settings.data.train_path)
    summary["figure_generation_enabled"] = plt is not None

    plot_target_balance(
        training_data,
        target_column=settings.training.target_column,
        output_path=args.figures_dir / "target_balance.png",
    )
    plot_missingness(
        training_data,
        output_path=args.figures_dir / "missingness.png",
    )
    plot_selected_distributions(
        training_data,
        output_path=args.figures_dir / "selected_distributions.png",
    )
    plot_delinquency_counts(
        training_data,
        output_path=args.figures_dir / "delinquency_distributions.png",
    )

    args.summary_output.write_text(
        json.dumps(_json_ready(summary), indent=2),
        encoding="utf-8",
    )

    print(f"EDA summary: {args.summary_output}")
    if plt is None:
        print("EDA figures: skipped because matplotlib is not installed")
    else:
        print(f"EDA figures: {args.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
