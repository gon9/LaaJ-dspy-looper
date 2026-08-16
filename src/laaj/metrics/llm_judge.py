"""LLM-as-a-Judge メトリクス"""

from collections.abc import Callable
from typing import Any

import dspy


class JudgeSignature(dspy.Signature):
    """評価基準に基づいて、予測された回答(prediction)が期待される正解(gold_answer)に対して適切かを厳正に判定する。"""

    criterion: str = dspy.InputField(desc="評価基準・指示")
    gold_answer: str = dspy.InputField(desc="正解データ・期待される回答")
    prediction: str = dspy.InputField(desc="モデルの予測出力")
    reasoning: str = dspy.OutputField(desc="判定に至った論理的な評価理由")
    score: float = dspy.OutputField(desc="0.0から1.0までの評価スコア（1.0が完全合格）")


class LLMJudgeMetric:
    """LLM-as-a-Judge 評価器"""

    def __init__(
        self,
        criterion: str = "期待される回答の内容・意図を満たしているかを判定してください。",
        threshold: float = 0.7,
        field_gold: str = "label",
        field_pred: str = "label",
    ):
        self.criterion = criterion
        self.threshold = threshold
        self.field_gold = field_gold
        self.field_pred = field_pred
        self.judge = dspy.ChainOfThought(JudgeSignature)

    def evaluate(self, example: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> float:
        """評価を実行

        Args:
            example: 正解データ
            pred: モデルの予測結果
            trace: 学習時の実行トレース

        Returns:
            float: 評価スコア
        """
        gold_val = str(getattr(example, self.field_gold, ""))
        pred_val = str(getattr(pred, self.field_pred, ""))

        try:
            result = self.judge(
                criterion=self.criterion,
                gold_answer=gold_val,
                prediction=pred_val,
            )
            score = float(result.score)
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.0

        if trace is not None:
            # 学習時 (Optimizer探索中): 連続値スコア（損失関数的役割）
            return score

        # 評価時 (Validation/Test): 閾値判定 (0.0 または 1.0)
        return 1.0 if score >= self.threshold else 0.0


def create_llm_judge_metric(
    criterion: str = "期待される回答の内容・意図を満たしているかを判定してください。",
    threshold: float = 0.7,
    field_gold: str = "label",
    field_pred: str = "label",
) -> Callable[[dspy.Example, dspy.Prediction, Any], float]:
    """LLM-as-a-Judge メトリクス関数を作成"""
    judge_instance = LLMJudgeMetric(
        criterion=criterion,
        threshold=threshold,
        field_gold=field_gold,
        field_pred=field_pred,
    )
    return judge_instance.evaluate
