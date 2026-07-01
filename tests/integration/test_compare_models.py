from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPARE_MODELS_SCRIPT = PROJECT_ROOT / "scripts" / "compare_models.py"


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
