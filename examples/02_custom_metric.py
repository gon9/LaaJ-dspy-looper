"""02_custom_metric.py - カスタム評価関数（Metric）の設計と評価

本スクリプトでは以下を学びます：
1. dspy.Example の作成と with_inputs() の役割
2. DSPy における metric 関数のシグネチャ `def metric(example, pred, trace=None)`
3. 学習時（trace is not None）と評価時（trace is None）での挙動の切り替え
4. dspy.Evaluate を用いたデータセット全体の自動評価
"""

import dspy


# 1. 評価対象のSignature & Module
class SentimentClassifier(dspy.Signature):
    """入力されたカスタマーレビューの感情を positive / negative / neutral に分類する。"""

    review_text: str = dspy.InputField(desc="顧客レビュー")
    sentiment: str = dspy.OutputField(desc="positive, negative, neutral のいずれか")


# 2. カスタムMetric関数の設計
def classification_metric(example, pred, trace=None) -> float:
    """分類タスクの評価関数。

    Args:
        example: 正解データ（dspy.Example）
        pred: モデルの予測結果
        trace: 学習時(Optimizer探索中)に渡される実行トレース情報。
               None の場合は通常の評価時（テスト/検証）。

    Returns:
        float: スコア（0.0 〜 1.0）
    """
    # 予測値と正解ラベルの正規化
    predicted_sentiment = getattr(pred, "sentiment", "").strip().lower()
    gold_sentiment = getattr(example, "expected_sentiment", "").strip().lower()

    # 完全一致チェック
    is_exact_match = predicted_sentiment == gold_sentiment

    if trace is not None:
        # [学習時 (Optimizer探索中)]:
        # 損失関数として機能するため、部分点（部分一致や関連度など）を返すと探索効率が上がります。
        if is_exact_match:
            return 1.0
        elif gold_sentiment in predicted_sentiment:
            # 形式崩れなどで余計な文字が入っているが正解を含んでいる場合は部分点 0.5
            return 0.5
        else:
            return 0.0
    else:
        # [評価時 (Validation/Test)]:
        # 厳密な判定スコア（1.0 または 0.0）を返す
        return 1.0 if is_exact_match else 0.0


def main():
    # 3. テストデータの準備 (dspy.Example)
    # 重要: `with_inputs("review_text")` で、どのフィールドを入力としてモデルに渡すかを明示します。
    dataset = [
        dspy.Example(
            review_text="この製品は素晴らしい！使いやすくて大満足です。",
            expected_sentiment="positive",
        ).with_inputs("review_text"),
        dspy.Example(
            review_text="初期不良ですぐに壊れました。サポート対応も最悪。",
            expected_sentiment="negative",
        ).with_inputs("review_text"),
        dspy.Example(
            review_text="普通の品質です。可もなく不可もなくといったところ。",
            expected_sentiment="neutral",
        ).with_inputs("review_text"),
        dspy.Example(
            review_text="デザインは良いが、値段が高すぎるのが難点。",
            expected_sentiment="negative",
        ).with_inputs("review_text"),
    ]

    print(f"作成したデータセット件数: {len(dataset)}件")
    print(f"データセット例 1: {dataset[0]}")
    print(f"入力フィールド定義: {dataset[0].inputs().keys()}\n")

    # 4. 評価のシミュレーション（ダミー予測オブジェクトでのスコア計算）
    class DummyPrediction:
        def __init__(self, sentiment):
            self.sentiment = sentiment

    print("=== Metric関数の単体テスト ===")
    pred_correct = DummyPrediction("positive")
    pred_partial = DummyPrediction("Label: positive (great product)")
    pred_wrong = DummyPrediction("negative")

    print(
        f"正解例 vs 正解予測 (評価時 trace=None) -> score: {classification_metric(dataset[0], pred_correct)}"
    )
    print(
        f"正解例 vs 部分一致予測 (評価時 trace=None) -> score: {classification_metric(dataset[0], pred_partial)}"
    )
    print(
        f"正解例 vs 部分一致予測 (学習時 trace!=None) -> score: {classification_metric(dataset[0], pred_partial, trace=True)}"
    )
    print(
        f"正解例 vs 不正解予測 (評価時 trace=None) -> score: {classification_metric(dataset[0], pred_wrong)}"
    )

    # 5. Evaluate クラスの使い方説明
    print("\n=== dspy.Evaluate の使用イメージ ===")
    print("dspy.Evaluate を使うと、データセット全体に対する平均スコアを並列・自動で集計できます:")
    print("""
    evaluator = Evaluate(
        devset=dataset,
        metric=classification_metric,
        num_threads=4,
        display_progress=True,
        display_table=5,
    )
    score = evaluator(module)
    print(f"Total Accuracy: {score}%")
    """)


if __name__ == "__main__":
    main()
