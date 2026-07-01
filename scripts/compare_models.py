#!/usr/bin/env python3
"""Build a persisted model-comparison report and champion rationale."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare persisted model artifacts and export the champion rationale."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "comparison",
        help="Directory where machine-readable comparison outputs will be written.",
    )
    parser.add_argument(
        "--markdown-path",
        type=Path,
        default=PROJECT_ROOT / "docs" / "MODEL_COMPARISON.md",
        help="Path where the human-readable comparison markdown will be written.",
    )
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _render_markdown(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    model_records = payload["model_records"]
    record_lookup = {record["model_key"]: record for record in model_records}
    champion = record_lookup[decision["champion_model_key"]]

    lines = [
        "# Model Comparison",
        "",
        "This document summarizes the current baseline, XGBoost, and PyTorch challenger results "
        "from persisted local artifacts and records the current champion rationale.",
        "",
        "## Current Ranking",
        "",
        "| Rank | Model | Validation PR-AUC | Validation ROC-AUC | Test PR-AUC | Test ROC-AUC | Test Brier |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in decision["comparison_table"]:
        lines.append(
            "| "
            f"{row['rank']} | {row['label']} | "
            f"{row['validation_average_precision']:.4f} | "
            f"{row['validation_roc_auc']:.4f} | "
            f"{row['test_average_precision']:.4f} | "
            f"{row['test_roc_auc']:.4f} | "
            f"{row['test_brier_score']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Current Champion",
            "",
            f"The current champion remains **{decision['champion_label']}**.",
            "",
            "Champion rationale:",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in decision["champion_reasoning"])

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            f"- {champion['label']} is the current deployment-ready path because it combines "
            "the strongest ranking performance with persisted threshold governance and active "
            "support in the local interactive and batch inference workflows.",
            "- The PyTorch challenger is useful as a non-tree benchmark, but its current raw-probability "
            "calibration and Brier performance lag the XGBoost path even though recall is higher at the default threshold.",
            "- The logistic-regression baseline remains the simplest benchmark and a useful reference point, "
            "but it is not competitive enough to replace the current champion.",
            "",
            "## Artifact Sources",
            "",
            "- `artifacts/baseline/baseline_metrics.json`",
            "- `artifacts/xgboost/xgboost_metrics.json`",
            "- `artifacts/xgboost/threshold_selection_report.json`",
            "- `artifacts/xgboost/cost_analysis_report.json`",
            "- `artifacts/xgboost/calibration_report.json`",
            "- `artifacts/torch/torch_metrics.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        from ml_risk_control.evaluation.comparison import build_default_comparison_payload
    except ImportError as error:
        print(
            "ERROR: Missing comparison dependencies. Install the project requirements first.",
            file=sys.stderr,
        )
        print(f"DETAIL: {error}", file=sys.stderr)
        return 1

    payload = build_default_comparison_payload(PROJECT_ROOT)
    payload["generated_at_utc"] = datetime.now(UTC).isoformat()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "model_comparison.json"
    rationale_path = args.output_dir / "champion_rationale.json"

    _write_json(json_path, payload)
    _write_json(
        rationale_path,
        {
            "generated_at_utc": payload["generated_at_utc"],
            "decision": payload["decision"],
        },
    )

    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.write_text(_render_markdown(payload), encoding="utf-8")

    print("Model comparison completed.")
    print(f"Comparison JSON: {json_path}")
    print(f"Champion rationale JSON: {rationale_path}")
    print(f"Markdown summary: {args.markdown_path}")
    print(f"Champion: {payload['decision']['champion_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
