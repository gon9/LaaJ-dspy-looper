"""03_bootstrap_fewshot.py - BootstrapFewShot による Few-Shot 自動最適化

本スクリプトでは以下を学びます：
1. BootstrapFewShot Optimizerの仕組み（成功トレースからデモ例を自動抽出）
2. 最適化前のベースライン評価とコンパイル
3. コンパイル済みモデルのFew-Shotデモ例の確認
4. 最適化済みModuleの永続化 (save / load)
"""

import os

import dspy
from dspy.teleprompt import BootstrapFewShot


# 1. タスクの定義: 経費精算の承認/否認判定
class ExpenseApproval(dspy.Signature):
    """提出された経費申請内容を審査し、社内規程に基づいて承認(approved)または否認(rejected)を判定する。"""

    expense_details: str = dspy.InputField(desc="経費申請の詳細（品目、金額、理由等）")
    approval_status: str = dspy.OutputField(desc="approved または rejected")


class ExpenseApprovalModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(ExpenseApproval)

    def forward(self, expense_details: str):
        return self.prog(expense_details=expense_details)


# 2. 評価関数の定義
def approval_metric(example, pred, trace=None) -> float:
    gold = getattr(example, "expected_status", "").strip().lower()
    predicted = getattr(pred, "approval_status", "").strip().lower()
    return 1.0 if gold == predicted else 0.0


def main():
    api_key = os.getenv("OPENAI_API_KEY")

    # 3. 少量データセットの準備（学習用と検証用）
    trainset = [
        dspy.Example(
            expense_details="クライアントとの懇親会飲食代 15,000円 (参加者3名、事前申請済)",
            expected_status="approved",
        ).with_inputs("expense_details"),
        dspy.Example(
            expense_details="個人の私用スマートフォンのゲーム課金 3,000円",
            expected_status="rejected",
        ).with_inputs("expense_details"),
        dspy.Example(
            expense_details="技術カンファレンス参加費 10,000円 (業務関連、領収書添付あり)",
            expected_status="approved",
        ).with_inputs("expense_details"),
        dspy.Example(
            expense_details="高級スパ利用代 50,000円 (リフレッシュ目的)",
            expected_status="rejected",
        ).with_inputs("expense_details"),
    ]

    devset = [
        dspy.Example(
            expense_details="出張時の新幹線特急券 14,000円 (東京-新大阪)",
            expected_status="approved",
        ).with_inputs("expense_details"),
        dspy.Example(
            expense_details="深夜の自宅からのゲームソフト購入 8,000円",
            expected_status="rejected",
        ).with_inputs("expense_details"),
    ]

    print("=== BootstrapFewShot 最適化パイプライン ===")
    print(f"訓練データ: {len(trainset)}件, 検証データ: {len(devset)}件")

    if not api_key:
        print("\n[Notice] OPENAI_API_KEY が未設定です。")
        print(
            "以下のコードで Optimizer を実行すると、DSPyが自動で正解トレースからFew-Shot例を生成・注入します:\n"
        )
        print("""
    # 1. LMの設定
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    # 2. 未学習モジュール（ベースライン）
    uncompiled_module = ExpenseApprovalModule()

    # 3. BootstrapFewShot Optimizerの初期化
    optimizer = BootstrapFewShot(
        metric=approval_metric,
        max_bootstrapped_demos=2,  # 自動生成するFew-Shot例の最大数
        max_labeled_demos=2,       # 正解ラベル付き例の最大数
    )

    # 4. コンパイル（プロンプト最適化の実行）
    compiled_module = optimizer.compile(student=uncompiled_module, trainset=trainset)

    # 5. 最適化結果の保存
    compiled_module.save("outputs/expense_approval_fewshot.json")
        """)
        return

    # APIキーがある場合の実際の最適化実行
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    # ベースライン評価
    base_module = ExpenseApprovalModule()
    evaluator = dspy.evaluate.Evaluate(devset=devset, metric=approval_metric, display_table=False)
    print("\n--- 最適化前（ベースライン）の評価 ---")
    base_score = evaluator(base_module)
    print(f"Base Accuracy: {base_score:.2f}%")

    # BootstrapFewShot の実行
    print("\n--- BootstrapFewShot によるコンパイル中 ---")
    optimizer = BootstrapFewShot(
        metric=approval_metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
    )
    compiled_module = optimizer.compile(student=base_module, trainset=trainset)

    # 最適化後の評価
    print("\n--- 最適化後の評価 ---")
    compiled_score = evaluator(compiled_module)
    print(f"Optimized Accuracy: {compiled_score:.2f}%")

    # 最適化済みモジュールの保存
    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/expense_approval_fewshot.json"
    compiled_module.save(save_path)
    print(f"\n最適化済みModuleを保存しました: {save_path}")


if __name__ == "__main__":
    main()
