"""Evaluation metrics for binary credit-risk classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _to_numeric_series(values: pd.Series | list[float] | np.ndarray, *, name: str) -> pd.Series:
    series = pd.Series(values, copy=False)
    numeric = pd.to_numeric(series, errors="raise")
    numeric.name = name
    return numeric.reset_index(drop=True)


def _validate_binary_target(y_true: pd.Series) -> None:
    distinct_values = set(y_true.dropna().unique().tolist())
    if not distinct_values.issubset({0, 1}):
        msg = "Binary evaluation requires y_true to contain only 0/1 values."
        raise ValueError(msg)
    if y_true.isna().any():
        msg = "Binary evaluation does not allow missing target values."
        raise ValueError(msg)


def _validate_probability_scores(y_score: pd.Series) -> None:
    if y_score.isna().any():
        msg = "Binary evaluation does not allow missing probability scores."
        raise ValueError(msg)
    if ((y_score < 0.0) | (y_score > 1.0)).any():
        msg = "Probability scores must be bounded within [0, 1]."
        raise ValueError(msg)


def compute_ks_statistic(
    y_true: pd.Series | list[int] | np.ndarray,
    y_score: pd.Series | list[float] | np.ndarray,
) -> float:
    """Compute the Kolmogorov-Smirnov separation statistic for binary scores."""
    target = _to_numeric_series(y_true, name="y_true").astype(int)
    scores = _to_numeric_series(y_score, name="y_score").astype(float)
    _validate_binary_target(target)
    _validate_probability_scores(scores)

    if len(target) != len(scores):
        msg = "y_true and y_score must have the same number of rows."
        raise ValueError(msg)

    distinct_classes = set(target.unique().tolist())
    if distinct_classes != {0, 1}:
        msg = "KS statistic requires both negative and positive classes to be present."
        raise ValueError(msg)

    evaluation_frame = pd.DataFrame({"y_true": target, "y_score": scores}).sort_values(
        by="y_score",
        ascending=False,
        kind="mergesort",
    )
    positives = int((evaluation_frame["y_true"] == 1).sum())
    negatives = int((evaluation_frame["y_true"] == 0).sum())

    cumulative_positive_rate = (evaluation_frame["y_true"] == 1).cumsum() / positives
    cumulative_negative_rate = (evaluation_frame["y_true"] == 0).cumsum() / negatives
    return float((cumulative_positive_rate - cumulative_negative_rate).abs().max())


def build_confusion_matrix_payload(
    y_true: pd.Series | list[int] | np.ndarray,
    y_pred: pd.Series | list[int] | np.ndarray,
) -> dict[str, Any]:
    """Return count and normalized confusion-matrix representations."""
    target = _to_numeric_series(y_true, name="y_true").astype(int)
    predictions = _to_numeric_series(y_pred, name="y_pred").astype(int)
    _validate_binary_target(target)

    if len(target) != len(predictions):
        msg = "y_true and y_pred must have the same number of rows."
        raise ValueError(msg)

    distinct_predictions = set(predictions.dropna().unique().tolist())
    if not distinct_predictions.issubset({0, 1}):
        msg = "Binary evaluation requires y_pred to contain only 0/1 values."
        raise ValueError(msg)
    if predictions.isna().any():
        msg = "Binary evaluation does not allow missing predicted labels."
        raise ValueError(msg)

    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    tn, fp, fn, tp = matrix.ravel()
    return {
        "labels": [0, 1],
        "counts": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": matrix.astype(int).tolist(),
        },
        "normalized": {
            "matrix": normalized.tolist(),
            "rows": {
                "actual_0": normalized[0].tolist(),
                "actual_1": normalized[1].tolist(),
            },
        },
    }


def evaluate_binary_classifier(
    y_true: pd.Series | list[int] | np.ndarray,
    y_score: pd.Series | list[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute the Stage 3 baseline metric bundle for a binary classifier."""
    if not 0.0 <= threshold <= 1.0:
        msg = "threshold must be bounded within [0, 1]."
        raise ValueError(msg)

    target = _to_numeric_series(y_true, name="y_true").astype(int)
    scores = _to_numeric_series(y_score, name="y_score").astype(float)
    _validate_binary_target(target)
    _validate_probability_scores(scores)

    if len(target) != len(scores):
        msg = "y_true and y_score must have the same number of rows."
        raise ValueError(msg)

    predicted_labels = scores.ge(threshold).astype(int)
    confusion = build_confusion_matrix_payload(target, predicted_labels)
    positive_rate = float((target == 1).mean())

    metrics = {
        "row_count": int(len(target)),
        "positive_rate": positive_rate,
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(target, scores)),
        "roc_auc": float(roc_auc_score(target, scores)),
        "ks_statistic": compute_ks_statistic(target, scores),
        "brier_score": float(brier_score_loss(target, scores)),
        "accuracy": float(accuracy_score(target, predicted_labels)),
        "precision": float(precision_score(target, predicted_labels, zero_division=0)),
        "recall": float(recall_score(target, predicted_labels, zero_division=0)),
        "f1": float(f1_score(target, predicted_labels, zero_division=0)),
        "confusion_matrix": confusion,
    }
    return metrics
