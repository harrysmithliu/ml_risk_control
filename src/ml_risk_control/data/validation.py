"""Reusable validation logic for raw files and schema-constrained datasets."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from ml_risk_control.data.contracts import FileContract, RawDataContracts


def sha256_for_file(path: Path) -> str:
    """Return the SHA-256 checksum for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(file_hashes: dict[str, str]) -> str:
    """Combine per-file hashes into a stable dataset-level fingerprint."""
    digest = hashlib.sha256()
    for name, file_hash in sorted(file_hashes.items()):
        digest.update(f"{name}:{file_hash}".encode("utf-8"))
    return digest.hexdigest()


def summarize_target(series: pd.Series) -> dict[str, Any]:
    """Summarize the observed class balance for a binary target column."""
    counts = Counter(int(value) for value in series.dropna().tolist())
    total = int(series.notna().sum())
    summary: dict[str, Any] = {
        "distinct_values": sorted(counts.keys()),
        "counts": {str(key): int(value) for key, value in counts.items()},
        "positive_rate": None,
    }
    if total > 0:
        summary["positive_rate"] = counts.get(1, 0) / total
    return summary


def validate_exact_columns(
    actual: list[str],
    expected: list[str],
    optional: list[str] | None = None,
) -> dict[str, Any]:
    """Compare actual columns against the declared contract."""
    optional_columns = optional or []
    return {
        "matches_expected_order": actual == expected,
        "missing_columns": [column for column in expected if column not in actual],
        "unexpected_columns": [
            column
            for column in actual
            if column not in expected and column not in optional_columns
        ],
        "optional_columns_present": [column for column in actual if column in optional_columns],
    }


def validate_dataframe_contract(
    dataframe: pd.DataFrame,
    contract: FileContract,
    *,
    source: str,
) -> dict[str, Any]:
    """Validate an in-memory dataframe against its declarative contract."""
    result: dict[str, Any] = {
        "path": source,
        "exists": True,
        "errors": [],
        "warnings": [],
    }
    actual_columns = dataframe.columns.tolist()
    duplicate_columns = [name for name, count in Counter(actual_columns).items() if count > 1]
    column_validation = validate_exact_columns(
        actual_columns,
        list(contract.expected_columns),
        optional=list(contract.optional_columns),
    )

    result["row_count"] = int(len(dataframe))
    result["column_count"] = int(len(actual_columns))
    result["columns"] = actual_columns
    result["duplicate_columns"] = duplicate_columns
    result["column_validation"] = column_validation
    result["missing_value_counts"] = {
        column: int(value)
        for column, value in dataframe.isna().sum().sort_index().items()
    }

    if duplicate_columns:
        result["errors"].append("Duplicate column names detected.")
    if column_validation["missing_columns"]:
        result["errors"].append("Required columns are missing.")
    if column_validation["unexpected_columns"]:
        result["errors"].append("Unexpected columns are present.")

    if contract.id_column:
        if contract.id_column not in dataframe.columns:
            result["errors"].append(f"Identifier column '{contract.id_column}' is missing.")
        else:
            result["id_is_unique"] = bool(dataframe[contract.id_column].is_unique)
            if not result["id_is_unique"]:
                result["warnings"].append("Identifier column is not unique.")

    if contract.target_mode == "required":
        if contract.target_column not in dataframe.columns:
            result["errors"].append(f"Target column '{contract.target_column}' is missing.")
        else:
            distinct_values = set(dataframe[contract.target_column].dropna().unique().tolist())
            result["target_summary"] = summarize_target(dataframe[contract.target_column])
            result["target_is_binary"] = distinct_values.issubset({0, 1})
            if not result["target_is_binary"]:
                result["errors"].append("Target column contains values outside {0, 1}.")
    elif contract.target_mode == "optional_empty" and contract.target_column in dataframe.columns:
        if dataframe[contract.target_column].notna().any():
            result["errors"].append(
                "Target column contains values in a file that should be unlabeled."
            )
        else:
            result["warnings"].append(
                "Target placeholder column is present in the unlabeled test file."
            )

    if result["row_count"] == 0:
        result["errors"].append("File is empty.")

    return result


def validate_file_contract(path: Path, contract: FileContract) -> dict[str, Any]:
    """Validate a single file against its declarative contract."""
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "errors": [],
        "warnings": [],
    }
    if not path.exists():
        message = "File does not exist."
        if contract.required:
            result["errors"].append(message)
        else:
            result["warnings"].append(message)
        return result

    file_hash = sha256_for_file(path)
    file_size_bytes = path.stat().st_size
    dataframe = pd.read_csv(path)
    result = validate_dataframe_contract(dataframe, contract, source=str(path))
    result["sha256"] = file_hash
    result["file_size_bytes"] = file_size_bytes
    return result


def build_raw_validation_report(
    *,
    raw_data_dir: Path,
    backend: str,
    contracts: RawDataContracts,
) -> dict[str, Any]:
    """Validate the raw dataset bundle and return a combined report payload."""
    files = {
        name: validate_file_contract(raw_data_dir / contract.file_name, contract)
        for name, contract in contracts.as_dict().items()
    }
    file_hashes = {
        name: report["sha256"]
        for name, report in files.items()
        if report.get("sha256")
    }
    all_errors = [error for report in files.values() for error in report["errors"]]

    return {
        "status": "passed" if not all_errors else "failed",
        "backend": backend,
        "raw_data_dir": str(raw_data_dir),
        "dataset_fingerprint": dataset_fingerprint(file_hashes) if file_hashes else None,
        "files": files,
    }
