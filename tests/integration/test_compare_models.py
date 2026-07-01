from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPARE_MODELS_SCRIPT = PROJECT_ROOT / "scripts" / "compare_models.py"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics_payload(
    *,
    model_name: str,
    validation_ap: float,
    validation_roc: float,
    validation_ks: float,
    validation_brier: float,
    test_ap: float,
    test_roc: float,
    test_ks: float,
    test_brier: float,
) -> dict[str, object]:
    def partition(
        average_precision: float,
        roc_auc: float,
        ks_statistic: float,
        brier_score: float,
    ) -> dict[str, float]:
        return {
            "average_precision": average_precision,
            "roc_auc": roc_auc,
            "ks_statistic": ks_statistic,
            "brier_score": brier_score,
            "accuracy": 0.91,
            "precision": 0.42,
            "recall": 0.51,
            "f1": 0.46,
        }

    return {
        "model_name": model_name,
        "model_version": "0.1.0",
        "schema_version": "1.0.0",
        "partitions": {
            "validation": partition(
                validation_ap,
                validation_roc,
                validation_ks,
                validation_brier,
            ),
            "test": partition(
                test_ap,
                test_roc,
                test_ks,
                test_brier,
            ),
        },
    }


def _run_summary_payload(
    selected_candidate_source: str | None,
    *,
    classifier_class: str,
) -> dict[str, object]:
    return {
        "selected_candidate_source": selected_candidate_source,
        "training_summary": {
            "row_count": 105000,
            "positive_rate": 0.0668,
            "eval_row_count": 22500,
            "trained_at_utc": "2026-07-01T00:00:00+00:00",
            "classifier_class": classifier_class,
        },
    }


def _build_comparison_fixture(project_root: Path) -> None:
    artifact_root = project_root / "artifacts"

    _write_json(
        artifact_root / "baseline" / "baseline_metrics.json",
        _metrics_payload(
            model_name="logistic_regression_baseline",
            validation_ap=0.3224,
            validation_roc=0.8278,
            validation_ks=0.5130,
            validation_brier=0.0534,
            test_ap=0.3303,
            test_roc=0.8311,
            test_ks=0.5071,
            test_brier=0.0531,
        ),
    )
    _write_json(
        artifact_root / "baseline" / "run_summary.json",
        _run_summary_payload(
            None,
            classifier_class="LogisticRegression",
        ),
    )

    _write_json(
        artifact_root / "xgboost" / "xgboost_metrics.json",
        _metrics_payload(
            model_name="xgboost_credit_risk",
            validation_ap=0.4195,
            validation_roc=0.8711,
            validation_ks=0.5930,
            validation_brier=0.0485,
            test_ap=0.4122,
            test_roc=0.8717,
            test_ks=0.5899,
            test_brier=0.0485,
        ),
    )
    _write_json(
        artifact_root / "xgboost" / "run_summary.json",
        _run_summary_payload(
            "reference",
            classifier_class="XGBClassifier",
        ),
    )
    _write_json(
        artifact_root / "xgboost" / "threshold_selection_report.json",
        {
            "validation_selection": {
                "recommended_threshold": 0.19963,
            }
        },
    )
    _write_json(
        artifact_root / "xgboost" / "cost_analysis_report.json",
        {
            "validation_selection": {
                "recommended_threshold": 0.13705,
            }
        },
    )
    _write_json(
        artifact_root / "xgboost" / "calibration_report.json",
        {
            "calibration": {
                "method": "sigmoid",
            },
            "partitions": {
                "validation": {
                    "raw": {"brier_score": 0.0485},
                    "calibrated": {"brier_score": 0.0491},
                }
            },
        },
    )

    _write_json(
        artifact_root / "torch" / "torch_metrics.json",
        _metrics_payload(
            model_name="torch_mlp_challenger",
            validation_ap=0.3976,
            validation_roc=0.8723,
            validation_ks=0.5904,
            validation_brier=0.1494,
            test_ap=0.3959,
            test_roc=0.8692,
            test_ks=0.5847,
            test_brier=0.1494,
        ),
    )
    _write_json(
        artifact_root / "torch" / "run_summary.json",
        _run_summary_payload(
            None,
            classifier_class="TorchMLPClassifier",
        ),
    )


def _load_compare_models_module():
    spec = importlib.util.spec_from_file_location(
        "compare_models_script",
        COMPARE_MODELS_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compare_models_main_writes_expected_outputs(monkeypatch, tmp_path: Path) -> None:
    module = _load_compare_models_module()
    _build_comparison_fixture(tmp_path)
    output_dir = tmp_path / "comparison"
    markdown_path = tmp_path / "MODEL_COMPARISON.md"

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            output_dir=output_dir,
            markdown_path=markdown_path,
        ),
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)

    exit_code = module.main()

    assert exit_code == 0
    assert (output_dir / "model_comparison.json").exists()
    assert (output_dir / "champion_rationale.json").exists()
    assert markdown_path.exists()

    comparison_payload = json.loads((output_dir / "model_comparison.json").read_text())
    rationale_payload = json.loads((output_dir / "champion_rationale.json").read_text())
    markdown_text = markdown_path.read_text()

    assert comparison_payload["decision"]["champion_model_key"] == "xgboost"
    assert comparison_payload["decision"]["comparison_table"][0]["model_key"] == "xgboost"
    assert rationale_payload["decision"]["champion_label"] == "XGBoost Champion Candidate"
    assert "The current champion remains **XGBoost Champion Candidate**." in markdown_text
    assert "| Rank | Model | Validation PR-AUC |" in markdown_text
