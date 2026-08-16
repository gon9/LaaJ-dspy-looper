"""Optimizer Engineのテスト"""

from unittest.mock import MagicMock

import dspy
import pytest
from pydantic import ValidationError

from laaj.config import OptimizerConfig
from laaj.optimizer.engine import OptimizerEngine


def dummy_metric(example, pred, trace=None) -> float:
    return 1.0


def test_create_bootstrap_fewshot():
    """BootstrapFewShotの作成をテスト"""
    config = OptimizerConfig(
        name="BootstrapFewShot",
        params={"max_bootstrapped_demos": 2, "max_labeled_demos": 2},
    )
    optimizer = OptimizerEngine.create_optimizer(config, dummy_metric)
    assert optimizer is not None


def test_create_bootstrap_fewshot_random_search():
    """BootstrapFewShotWithRandomSearchの作成をテスト"""
    config = OptimizerConfig(
        name="BootstrapFewShotWithRandomSearch",
        params={
            "max_bootstrapped_demos": 2,
            "max_labeled_demos": 2,
            "num_candidate_programs": 2,
        },
    )
    optimizer = OptimizerEngine.create_optimizer(config, dummy_metric)
    assert optimizer is not None


def test_create_unknown_optimizer_validation():
    """未知のOptimizer指定時にPydanticバリデーションエラーが発生することをテスト"""
    with pytest.raises(ValidationError):
        OptimizerConfig(
            name="UnknownOptimizer",  # type: ignore
            params={},
        )


def test_create_unknown_optimizer_engine():
    """エンジンに直接未知のOptimizer名が渡された場合にValueErrorが発生することをテスト"""
    mock_config = MagicMock()
    mock_config.name = "UnsupportedOptimizer"
    mock_config.params = {}

    with pytest.raises(ValueError, match="サポートされていないOptimizer"):
        OptimizerEngine.create_optimizer(mock_config, dummy_metric)


def test_optimize_with_valset():
    """検証セット付きでの最適化実行をテスト"""
    mock_optimizer = MagicMock()
    mock_student = MagicMock(spec=dspy.Module)
    mock_trainset = [MagicMock(spec=dspy.Example)]
    mock_valset = [MagicMock(spec=dspy.Example)]

    mock_compiled_module = MagicMock(spec=dspy.Module)
    mock_optimizer.compile.return_value = mock_compiled_module

    result = OptimizerEngine.optimize(
        optimizer=mock_optimizer,
        student=mock_student,
        trainset=mock_trainset,
        valset=mock_valset,
    )

    mock_optimizer.compile.assert_called_once_with(
        student=mock_student, trainset=mock_trainset, valset=mock_valset
    )
    assert result == mock_compiled_module


def test_optimize_without_valset():
    """検証セットなしでの最適化実行をテスト"""
    mock_optimizer = MagicMock()
    mock_student = MagicMock(spec=dspy.Module)
    mock_trainset = [MagicMock(spec=dspy.Example)]

    mock_compiled_module = MagicMock(spec=dspy.Module)
    mock_optimizer.compile.return_value = mock_compiled_module

    result = OptimizerEngine.optimize(
        optimizer=mock_optimizer,
        student=mock_student,
        trainset=mock_trainset,
        valset=None,
    )

    mock_optimizer.compile.assert_called_once_with(student=mock_student, trainset=mock_trainset)
    assert result == mock_compiled_module
