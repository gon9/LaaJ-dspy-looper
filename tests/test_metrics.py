"""メトリクスのテスト"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import dspy
import pytest

from laaj.metrics import (
    MetricRegistry,
    exact_match_metric,
)
from laaj.metrics.llm_judge import LLMJudgeMetric
from laaj.metrics.semantic_similarity import create_semantic_similarity_metric


def test_exact_match_metric():
    """完全一致メトリクスのテスト"""
    metric = exact_match_metric(field="label")

    example = dspy.Example(text="sample text", label="positive")
    pred_correct = dspy.Prediction(label="positive")
    pred_incorrect = dspy.Prediction(label="negative")

    assert metric(example, pred_correct) == 1.0
    assert metric(example, pred_incorrect) == 0.0


def test_semantic_similarity_metric():
    """意味的類似度メトリクスのテスト"""
    metric = create_semantic_similarity_metric(
        field_gold="label", field_pred="label", threshold=0.5
    )

    example = dspy.Example(label="positive review")
    pred_exact = dspy.Prediction(label="positive review")
    pred_similar = dspy.Prediction(label="positive sentiment review")
    pred_different = dspy.Prediction(label="totally different negative content")

    # 完全一致
    assert metric(example, pred_exact) == 1.0

    # 類似テキスト（評価時）
    assert metric(example, pred_similar) == 1.0

    # 類似テキスト（学習時 trace!=None）
    trace_score = metric(example, pred_similar, trace=True)
    assert 0.0 < trace_score <= 1.0

    # 異質なテキスト
    assert metric(example, pred_different) == 0.0


def test_llm_judge_metric():
    """LLM-as-a-Judge メトリクスのテスト"""
    judge_metric = LLMJudgeMetric(
        criterion="テスト基準",
        threshold=0.7,
        field_gold="label",
        field_pred="label",
    )

    # judge内部のChainOfThoughtをモック
    mock_result = MagicMock()
    mock_result.score = 0.85
    mock_result.reasoning = "基準を満たしています"
    judge_metric.judge = MagicMock(return_value=mock_result)

    example = dspy.Example(label="approved")
    pred = dspy.Prediction(label="approved")

    # 評価時
    eval_score = judge_metric.evaluate(example, pred, trace=None)
    assert eval_score == 1.0

    # 学習時 (trace != None -> 連続スコア)
    train_score = judge_metric.evaluate(example, pred, trace=True)
    assert train_score == 0.85


def test_metric_registry_builtin():
    """組み込みメトリクスの取得をテスト"""
    metric = MetricRegistry.get_metric("exact_match", field="label")
    assert callable(metric)

    llm_metric = MetricRegistry.get_metric("llm_judge", field="label")
    assert callable(llm_metric)

    sim_metric = MetricRegistry.get_metric("semantic_similarity", field="label")
    assert callable(sim_metric)


def test_metric_registry_custom_metric():
    """カスタムメトリクスのロードをテスト"""
    with tempfile.TemporaryDirectory() as tmpdir:
        metric_file = Path(tmpdir) / "my_metric.py"
        metric_file.write_text(
            """
def custom_eval(example, pred, trace=None):
    return 1.0 if getattr(pred, 'out') == getattr(example, 'out') else 0.0
""",
            encoding="utf-8",
        )

        metric = MetricRegistry.get_metric(
            name="custom",
            custom_path=str(metric_file),
            custom_func="custom_eval",
        )
        assert callable(metric)

        ex = dspy.Example(out="ok")
        pr = dspy.Prediction(out="ok")
        assert metric(ex, pr) == 1.0


def test_metric_registry_missing_field():
    """fieldパラメータが不足している場合のエラーをテスト"""
    with pytest.raises(ValueError, match="field パラメータが必要です"):
        MetricRegistry.get_metric("exact_match")


def test_metric_registry_unknown_metric():
    """存在しないメトリクス名が指定された場合のエラーをテスト"""
    with pytest.raises(ValueError, match="メトリクス 'unknown_metric' が見つかりません"):
        MetricRegistry.get_metric("unknown_metric")
