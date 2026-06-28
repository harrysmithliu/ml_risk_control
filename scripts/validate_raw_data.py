#!/usr/bin/env python3
"""Validate the raw Give Me Some Credit dataset and emit a fingerprinted report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_risk_control.config import get_settings
from ml_risk_control.data.contracts import build_gmsc_raw_data_contracts
from ml_risk_control.data.validation import build_raw_validation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate raw Kaggle credit risk files and generate a JSON report."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "validation" / "raw_data_validation_report.json",
        help="Path to the JSON validation report.",
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    settings = get_settings()
    contracts = build_gmsc_raw_data_contracts(
        id_column=settings.training.id_column,
        target_column=settings.training.target_column,
        train_file_name=settings.data.train_file,
        test_file_name=settings.data.test_file,
    )
    report = build_raw_validation_report(
        raw_data_dir=settings.data.raw_data_dir,
        backend=settings.data.backend,
        contracts=contracts,
    )
    report["generated_at_utc"] = datetime.now(UTC).isoformat()
    all_errors = [error for file_report in report["files"].values() for error in file_report["errors"]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Validation status: {report['status']}")
    print(f"Validation report: {args.output}")
    if report["dataset_fingerprint"]:
        print(f"Dataset fingerprint: {report['dataset_fingerprint']}")

    if all_errors:
        for error in all_errors:
            print(f"ERROR: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
