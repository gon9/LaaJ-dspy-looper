"""メトリクスのテスト"""

import dspy
import pytest

from laaj.metrics import MetricRegistry, exact_match_metric


def test_exact_match_metric():
    """完全一致メトリクスのテスト"""
    metric = exact_match_metric(field="label")

    example = dspy.Example(label="A")
    pred_correct = dspy.Prediction(label="A")
    pred_incorrect = dspy.Prediction(label="B")

    assert metric(example, pred_correct) == 1.0
    assert metric(example, pred_incorrect) == 0.0


def test_metric_registry_builtin():
    """組み込みメトリクスの取得テスト"""
    metric = MetricRegistry.get_metric(name="exact_match", field="label")
    assert callable(metric)


def test_metric_registry_missing_field():
    """フィールド指定なしのテスト"""
    with pytest.raises(ValueError, match="field パラメータが必要です"):
        MetricRegistry.get_metric(name="exact_match")


def test_metric_registry_unknown_metric():
    """未知のメトリクスのテスト"""
    with pytest.raises(ValueError, match="メトリクス .* が見つかりません"):
        MetricRegistry.get_metric(name="unknown_metric", field="label")
