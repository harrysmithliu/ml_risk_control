from __future__ import annotations

import pytest

from ml_risk_control.evaluation.metrics import (
    build_confusion_matrix_payload,
    build_precision_recall_curve_payload,
    build_roc_curve_payload,
    compute_ks_statistic,
    evaluate_binary_classifier,
)


def test_compute_ks_statistic_returns_expected_value_for_separated_scores() -> None:
    y_true = [0, 0, 1, 1]
    y_score = [0.10, 0.20, 0.80, 0.90]

    ks_statistic = compute_ks_statistic(y_true, y_score)

    assert ks_statistic == pytest.approx(1.0)


def test_compute_ks_statistic_raises_when_only_one_class_is_present() -> None:
    y_true = [0, 0, 0]
    y_score = [0.10, 0.20, 0.30]

    with pytest.raises(ValueError, match="requires both negative and positive classes"):
        compute_ks_statistic(y_true, y_score)


def test_build_confusion_matrix_payload_returns_counts_and_normalized_rows() -> None:
    payload = build_confusion_matrix_payload(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 0, 1],
    )

    assert payload["labels"] == [0, 1]
    assert payload["counts"]["tn"] == 1
    assert payload["counts"]["fp"] == 1
    assert payload["counts"]["fn"] == 1
    assert payload["counts"]["tp"] == 1
    assert payload["counts"]["matrix"] == [[1, 1], [1, 1]]
    assert payload["normalized"]["rows"]["actual_0"] == pytest.approx([0.5, 0.5])
    assert payload["normalized"]["rows"]["actual_1"] == pytest.approx([0.5, 0.5])


def test_build_confusion_matrix_payload_rejects_non_binary_predictions() -> None:
    with pytest.raises(ValueError, match="y_pred to contain only 0/1 values"):
        build_confusion_matrix_payload(
            y_true=[0, 1, 0],
            y_pred=[0, 2, 1],
        )


def test_build_precision_recall_curve_payload_returns_plot_ready_arrays() -> None:
    payload = build_precision_recall_curve_payload(
        y_true=[0, 0, 1, 1],
        y_score=[0.10, 0.20, 0.80, 0.90],
    )

    assert payload["point_count"] == 5
    assert payload["threshold_count"] == 4
    assert payload["baseline_positive_rate"] == pytest.approx(0.5)
    assert payload["precision"][0] == pytest.approx(0.5)
    assert payload["recall"][0] == pytest.approx(1.0)
    assert payload["thresholds"] == pytest.approx([0.1, 0.2, 0.8, 0.9])


def test_build_roc_curve_payload_returns_plot_ready_arrays() -> None:
    payload = build_roc_curve_payload(
        y_true=[0, 0, 1, 1],
        y_score=[0.10, 0.20, 0.80, 0.90],
    )

    assert payload["point_count"] >= 3
    assert payload["threshold_count"] == payload["point_count"]
    assert payload["false_positive_rate"][0] == pytest.approx(0.0)
    assert payload["true_positive_rate"][-1] == pytest.approx(1.0)


def test_build_roc_curve_payload_rejects_single_class_targets() -> None:
    with pytest.raises(ValueError, match="requires both negative and positive classes"):
        build_roc_curve_payload(
            y_true=[1, 1, 1],
            y_score=[0.5, 0.6, 0.7],
        )


def test_evaluate_binary_classifier_returns_stage_3_metric_bundle() -> None:
    y_true = [0, 0, 1, 1, 0, 1]
    y_score = [0.10, 0.30, 0.80, 0.90, 0.40, 0.70]

    metrics = evaluate_binary_classifier(y_true, y_score, threshold=0.5)

    assert metrics["row_count"] == 6
    assert metrics["positive_rate"] == pytest.approx(0.5)
    assert metrics["threshold"] == 0.5
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["ks_statistic"] == pytest.approx(1.0)
    assert metrics["brier_score"] == pytest.approx(0.06666666666666667)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"]["counts"]["matrix"] == [[3, 0], [0, 3]]
    assert metrics["precision_recall_curve"]["baseline_positive_rate"] == pytest.approx(0.5)
    assert metrics["roc_curve"]["false_positive_rate"][0] == pytest.approx(0.0)


def test_evaluate_binary_classifier_respects_threshold_for_predicted_labels() -> None:
    y_true = [0, 1, 1, 0]
    y_score = [0.40, 0.55, 0.60, 0.45]

    metrics = evaluate_binary_classifier(y_true, y_score, threshold=0.6)

    assert metrics["confusion_matrix"]["counts"]["tn"] == 2
    assert metrics["confusion_matrix"]["counts"]["fp"] == 0
    assert metrics["confusion_matrix"]["counts"]["fn"] == 1
    assert metrics["confusion_matrix"]["counts"]["tp"] == 1
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)


def test_evaluate_binary_classifier_rejects_probability_scores_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="bounded within \\[0, 1\\]"):
        evaluate_binary_classifier(
            y_true=[0, 1, 0],
            y_score=[0.10, 1.20, 0.30],
        )


def test_evaluate_binary_classifier_rejects_threshold_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="threshold must be bounded within \\[0, 1\\]"):
        evaluate_binary_classifier(
            y_true=[0, 1, 0],
            y_score=[0.10, 0.80, 0.30],
            threshold=1.5,
        )
