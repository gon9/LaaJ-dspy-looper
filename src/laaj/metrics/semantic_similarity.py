"""Semantic Similarity メトリクス"""

import re
from collections.abc import Callable
from typing import Any

import dspy


def _tokenize(text: str) -> set[str]:
    """テキストを単語・文字トークン集合に分割"""
    # 簡易トークナイズ（日本語文字や英単語を抽出）
    tokens = re.findall(r"[\w]+", text.lower())
    # 1文字単位のN-gramも生成（日本語文字列の類似度向上）
    char_ngrams = [text[i : i + 2] for i in range(max(0, len(text) - 1))]
    return set(tokens) | set(char_ngrams)


def _compute_jaccard_similarity(str1: str, str2: str) -> float:
    """Jaccard 係数による類似度を計算"""
    set1 = _tokenize(str1)
    set2 = _tokenize(str2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def create_semantic_similarity_metric(
    field_gold: str = "label",
    field_pred: str = "label",
    threshold: float = 0.6,
) -> Callable[[dspy.Example, dspy.Prediction, Any], float]:
    """意味的類似度（Semantic Similarity）メトリクス関数を作成

    Args:
        field_gold: 正解データのフィールド名
        field_pred: 予測データのフィールド名
        threshold: 評価判定用の類似度閾値

    Returns:
        Callable: 評価関数
    """

    def metric(example: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> float:
        gold_val = str(getattr(example, field_gold, "")).strip()
        pred_val = str(getattr(pred, field_pred, "")).strip()

        if gold_val == pred_val:
            return 1.0

        similarity = _compute_jaccard_similarity(gold_val, pred_val)

        if trace is not None:
            # 学習時: 類似度スコアそのものを返す
            return float(similarity)

        # 評価時: 閾値を超えていれば合格
        return 1.0 if similarity >= threshold else 0.0

    return metric
