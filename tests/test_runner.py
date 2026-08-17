"""Runner層の単体・結合テスト"""

from unittest.mock import MagicMock, patch

import pytest

from laaj.config import load_config
from laaj.exceptions import ConfigError
from laaj.modules import ModuleRegistry
from laaj.runner import (
    DiffRunner,
    EvaluationRunner,
    ExportRunner,
    OptimizationRunner,
)


@pytest.fixture
def sample_config():
    return load_config("experiments/example.yaml")


@patch("laaj.runner.optimization.LMFactory.configure_default_lm")
@patch("laaj.runner.optimization.OptimizerEngine.optimize")
@patch("dspy.Evaluate")
def test_optimization_runner_run(mock_eval_cls, mock_optimize, mock_lm, sample_config, tmp_path):
    """OptimizationRunnerの実行フローをテスト"""
    mock_module = MagicMock()
    mock_optimize.return_value = mock_module

    mock_evaluator = MagicMock()
    mock_evaluator.return_value = 0.95
    mock_eval_cls.return_value = mock_evaluator

    runner = OptimizationRunner(sample_config)
    messages = []
    result = runner.run(progress_callback=messages.append)

    assert result.test_score == 0.95
    assert result.experiment_name == sample_config.experiment.name
    assert result.train_size > 0
    assert len(messages) > 0
    mock_module.save.assert_called_once()


@patch("laaj.runner.evaluation.LMFactory.configure_default_lm")
@patch("dspy.Evaluate")
def test_evaluation_runner_baseline(mock_eval_cls, mock_lm, sample_config):
    """EvaluationRunnerのbaseline評価をテスト"""
    mock_evaluator = MagicMock()
    mock_evaluator.return_value = 0.82
    mock_eval_cls.return_value = mock_evaluator

    runner = EvaluationRunner(sample_config)
    result = runner.run(baseline=True)

    assert result.test_score == 0.82
    assert result.module_type == "baseline"


def test_evaluation_runner_missing_args(sample_config):
    """baselineもmodule_pathも指定しない場合にエラーとなることをテスト"""
    runner = EvaluationRunner(sample_config)
    with pytest.raises(ConfigError):
        runner.run(baseline=False, module_path=None)


@patch("laaj.runner.diff_export.ModuleInspector.diff_modules")
def test_diff_runner(mock_diff, sample_config, tmp_path):
    """DiffRunnerの実行をテスト"""
    mock_diff.return_value = {"diffs": []}
    dummy_module = tmp_path / "mod.json"
    temp_mod = ModuleRegistry.instantiate_module(
        sample_config.module.path, sample_config.module.class_name
    )
    temp_mod.save(str(dummy_module))

    runner = DiffRunner(sample_config)
    res = runner.run(module_path=dummy_module)
    assert "diffs" in res


@patch("laaj.runner.diff_export.ModuleInspector.export_module")
def test_export_runner(mock_export, sample_config, tmp_path):
    """ExportRunnerの実行をテスト"""
    mock_export.return_value = '{"test": 123}'
    dummy_module = tmp_path / "mod.json"
    dummy_module.write_text("{}", encoding="utf-8")
    out_file = tmp_path / "out.json"

    runner = ExportRunner(sample_config)
    res = runner.run(module_path=dummy_module, output_format="json", output_file=out_file)

    assert '{"test": 123}' in res
    assert out_file.read_text(encoding="utf-8") == '{"test": 123}'
