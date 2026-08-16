"""エクスポート・差分比較のテスト"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import dspy

from laaj.export.exporter import ModuleInspector
from laaj.modules.registry import ModuleRegistry


def test_extract_info():
    """モジュール情報の抽出をテスト"""
    mock_module = MagicMock(spec=dspy.Module)
    mock_pred = MagicMock()
    mock_sig = MagicMock()
    mock_sig.instructions = "Sample instruction"
    mock_pred.signature = mock_sig
    mock_pred.demos = [{"input": "test", "output": "positive"}]

    mock_module.named_predictors.return_value = [("classifier", mock_pred)]

    info = ModuleInspector.extract_info(mock_module)
    assert len(info["predictors"]) == 1
    assert info["predictors"][0]["name"] == "classifier"
    assert info["predictors"][0]["instructions"] == "Sample instruction"
    assert info["predictors"][0]["demos_count"] == 1


def test_diff_modules():
    """モジュール差分の比較をテスト"""
    base_module = MagicMock(spec=dspy.Module)
    base_pred = MagicMock()
    base_sig = MagicMock()
    base_sig.instructions = "Base instruction"
    base_pred.signature = base_sig
    base_pred.demos = []
    base_module.named_predictors.return_value = [("classifier", base_pred)]

    opt_module = MagicMock(spec=dspy.Module)
    opt_pred = MagicMock()
    opt_sig = MagicMock()
    opt_sig.instructions = "Optimized instruction"
    opt_pred.signature = opt_sig
    opt_pred.demos = [{"text": "sample", "label": "positive"}]
    opt_module.named_predictors.return_value = [("classifier", opt_pred)]

    diff = ModuleInspector.diff_modules(base_module, opt_module)
    assert len(diff["diffs"]) == 1
    d = diff["diffs"][0]
    assert d["predictor_name"] == "classifier"
    assert d["instruction_changed"] is True
    assert d["base_instruction"] == "Base instruction"
    assert d["optimized_instruction"] == "Optimized instruction"
    assert d["demos_added"] == 1


def test_export_module_json_and_markdown():
    """JSONおよびMarkdown形式のエクスポートをテスト"""
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_save = Path(tmpdir) / "module.json"
        temp_mod = ModuleRegistry.instantiate_module(
            "examples/simple_classifier.py", "SimpleClassifier"
        )
        temp_mod.save(str(dummy_save))

        # JSON形式
        json_output = ModuleInspector.export_module(
            module_path="examples/simple_classifier.py",
            class_name="SimpleClassifier",
            saved_path=str(dummy_save),
            output_format="json",
        )
        parsed = json.loads(json_output)
        assert "predictors" in parsed

        # Markdown形式
        md_output = ModuleInspector.export_module(
            module_path="examples/simple_classifier.py",
            class_name="SimpleClassifier",
            saved_path=str(dummy_save),
            output_format="markdown",
        )
        assert "# Optimized DSPy Module Export" in md_output
