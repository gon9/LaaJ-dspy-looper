"""完全一致メトリクス"""

import dspy


def exact_match_metric(field: str):
    """完全一致メトリクスを生成

    Args:
        field: 比較対象のフィールド名

    Returns:
        callable: メトリクス関数
    """

    def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        """完全一致を評価

        Args:
            example: 正解データ
            pred: 予測結果
            trace: トレース情報（オプション）

        Returns:
            float: 1.0（一致）または0.0（不一致）
        """
        expected = getattr(example, field, None)
        predicted = getattr(pred, field, None)

        if expected is None or predicted is None:
            return 0.0

        is_match = str(expected).strip() == str(predicted).strip()
        return float(is_match)

    return metric
