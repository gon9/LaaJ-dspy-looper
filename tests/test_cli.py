"""CLIコマンドのテスト"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from laaj.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_help(runner):
    """CLIのヘルプコマンドをテスト"""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "LaaJ-dspy-looper" in result.output
    assert "optimize" in result.output
    assert "evaluate" in result.output


def test_evaluate_missing_flag(runner):
    """--baselineも--module-pathも指定しない場合にメッセージを出力して終了することをテスト"""
    result = runner.invoke(
        cli,
        [
            "evaluate",
            "--config",
            "experiments/example.yaml",
        ],
        env={"OPENAI_API_KEY": "dummy_key"},
    )
    assert result.exit_code == 0
    assert "--baseline または --module-path のいずれかを指定してください" in result.output


@patch("laaj.cli._configure_lm")
@patch("laaj.cli._evaluate_module")
@patch("laaj.cli.LocalLogger.log_evaluation")
def test_evaluate_baseline(mock_log, mock_eval, mock_cfg_lm, runner):
    """--baseline での評価実行をテスト"""
    mock_eval.return_value = 0.85
    mock_log.return_value = "outputs/example_classification/eval_log.json"

    result = runner.invoke(
        cli,
        [
            "evaluate",
            "--config",
            "experiments/example.yaml",
            "--baseline",
        ],
        env={"OPENAI_API_KEY": "dummy_key"},
    )

    assert result.exit_code == 0
    assert "スコア (baseline): 0.8500" in result.output
    mock_eval.assert_called_once()


@patch("laaj.cli._configure_lm")
@patch("laaj.cli.OptimizerEngine.optimize")
@patch("laaj.cli._evaluate_module")
@patch("laaj.cli.LocalLogger.log_experiment")
def test_optimize_command(mock_log, mock_eval, mock_opt, mock_cfg_lm, runner):
    """optimize コマンドの実行フローをテスト"""
    mock_module = MagicMock()
    mock_opt.return_value = mock_module
    mock_eval.return_value = 0.92
    mock_log.return_value = "outputs/example_classification/experiment_log.json"

    result = runner.invoke(
        cli,
        [
            "optimize",
            "--config",
            "experiments/example.yaml",
        ],
        env={"OPENAI_API_KEY": "dummy_key"},
    )

    assert result.exit_code == 0
    assert "最適化完了" in result.output
    assert "テストスコア: 0.9200" in result.output
    mock_opt.assert_called_once()
    mock_module.save.assert_called_once()


@patch("laaj.cli.ModuleRegistry.instantiate_module")
@patch("laaj.cli.ModuleInspector.diff_modules")
def test_diff_command(mock_diff, mock_instantiate, runner, tmp_path):
    """diff コマンドの実行をテスト"""
    mock_diff.return_value = {
        "diffs": [
            {
                "predictor_name": "classifier",
                "instruction_changed": True,
                "base_instruction": "Base prompt",
                "optimized_instruction": "Optimized prompt",
                "base_demos_count": 0,
                "optimized_demos_count": 2,
                "demos_added": 2,
                "new_demos": [{"text": "sample", "label": "positive"}],
            }
        ]
    }
    dummy_module_file = tmp_path / "module.json"
    dummy_module_file.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        cli,
        [
            "diff",
            "--config",
            "experiments/example.yaml",
            "--module-path",
            str(dummy_module_file),
        ],
    )
    assert result.exit_code == 0
    assert "プロンプト最適化 差分レポート" in result.output
    assert "Base prompt" in result.output
    assert "Optimized prompt" in result.output


@patch("laaj.cli.ModuleInspector.export_module")
def test_export_command(mock_export, runner, tmp_path):
    """export コマンドの実行をテスト"""
    mock_export.return_value = '{"exported": true}'
    dummy_module_file = tmp_path / "module.json"
    dummy_module_file.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        cli,
        [
            "export",
            "--config",
            "experiments/example.yaml",
            "--module-path",
            str(dummy_module_file),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    assert '{"exported": true}' in result.output
