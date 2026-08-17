"""CLIコマンドのテスト"""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from laaj.cli import cli
from laaj.runner import EvaluationResult, OptimizationResult


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
    assert "diff" in result.output
    assert "export" in result.output


def test_evaluate_missing_flag(runner):
    """--baselineも--module-pathも指定しない場合にメッセージを出力することをテスト"""
    result = runner.invoke(
        cli,
        [
            "evaluate",
            "--config",
            "experiments/example.yaml",
        ],
    )
    assert result.exit_code == 0
    assert "--baseline または --module-path のいずれかを指定してください" in result.output


@patch("laaj.runner.evaluation.EvaluationRunner.run")
def test_evaluate_baseline(mock_eval_run, runner):
    """--baseline での評価実行をテスト"""
    mock_eval_run.return_value = EvaluationResult(
        experiment_name="example_classification",
        module_type="baseline",
        test_score=0.85,
        test_size=10,
        module_path=None,
        log_path="outputs/example_classification/eval.json",
    )

    result = runner.invoke(
        cli,
        [
            "evaluate",
            "--config",
            "experiments/example.yaml",
            "--baseline",
        ],
    )

    assert result.exit_code == 0
    assert "スコア (baseline) = 0.8500" in result.output
    mock_eval_run.assert_called_once()


@patch("laaj.runner.optimization.OptimizationRunner.run")
def test_optimize_command(mock_opt_run, runner):
    """optimize コマンドの実行フローをテスト"""
    mock_opt_run.return_value = OptimizationResult(
        experiment_name="example_classification",
        test_score=0.92,
        train_size=20,
        val_size=5,
        test_size=5,
        optimized_module_path="outputs/mod.json",
        log_path="outputs/log.json",
    )

    result = runner.invoke(
        cli,
        [
            "optimize",
            "--config",
            "experiments/example.yaml",
        ],
    )

    assert result.exit_code == 0
    assert "最適化が正常に完了しました" in result.output
    assert "0.9200" in result.output
    mock_opt_run.assert_called_once()


@patch("laaj.runner.diff_export.DiffRunner.run")
def test_diff_command(mock_diff_run, runner, tmp_path):
    """diff コマンドの実行をテスト"""
    mock_diff_run.return_value = {
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


@patch("laaj.runner.diff_export.ExportRunner.run")
def test_export_command(mock_export_run, runner, tmp_path):
    """export コマンドの実行をテスト"""
    mock_export_run.return_value = '{"exported": true}'
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


@patch("laaj.runner.optimization.OptimizationRunner.run")
def test_optimize_command_error(mock_opt_run, runner):
    """optimize コマンドでエラーが発生した場合の異常終了をテスト"""
    mock_opt_run.side_effect = Exception("Optimization failed")

    result = runner.invoke(
        cli,
        [
            "optimize",
            "--config",
            "experiments/example.yaml",
        ],
    )
    assert result.exit_code == 1
    assert "エラーが発生しました" in result.output


@patch("laaj.runner.evaluation.EvaluationRunner.run")
def test_evaluate_command_error(mock_eval_run, runner):
    """evaluate コマンドでエラーが発生した場合の異常終了をテスト"""
    mock_eval_run.side_effect = Exception("Evaluation failed")

    result = runner.invoke(
        cli,
        [
            "evaluate",
            "--config",
            "experiments/example.yaml",
            "--baseline",
        ],
    )
    assert result.exit_code == 1
    assert "エラーが発生しました" in result.output
