"""04_mipro_optimization.py - MIPROv2 による指示文（Instruction）自動最適化

本スクリプトでは以下を学びます：
1. MIPROv2 (Multiprompt Instruction PRoposal Optimizer v2) の概念
2. Meta-LLM による指示文（Instruction）候補の自動生成とベイズ探索
3. MIPROv2の主要ハイパーパラメータの役割
4. 最適化前後の指示文の変化の観察
"""

import os

import dspy
from dspy.teleprompt import MIPROv2


# 1. Signature & Module 定義
class CustomerSupportTriage(dspy.Signature):
    """ユーザーからの問い合わせ内容を分類し、優先度(priority: High/Medium/Low)と担当部署(category)を出力する。"""

    inquiry: str = dspy.InputField(desc="問い合わせ本文")
    priority: str = dspy.OutputField(desc="High, Medium, Low のいずれか")
    category: str = dspy.OutputField(desc="Billing, Technical, Account, General のいずれか")


class SupportTriageModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(CustomerSupportTriage)

    def forward(self, inquiry: str):
        return self.prog(inquiry=inquiry)


# 2. 複合Metric関数の定義（優先度とカテゴリの同時判定）
def triage_metric(example, pred, trace=None) -> float:
    pred_priority = getattr(pred, "priority", "").strip().lower()
    gold_priority = getattr(example, "expected_priority", "").strip().lower()

    pred_category = getattr(pred, "category", "").strip().lower()
    gold_category = getattr(example, "expected_category", "").strip().lower()

    priority_match = pred_priority == gold_priority
    category_match = pred_category == gold_category

    if priority_match and category_match:
        return 1.0

    if trace is not None:
        # 片方だけ合っている場合は部分点 0.5
        if priority_match or category_match:
            return 0.5
        return 0.0

    return 0.0


def main():
    api_key = os.getenv("OPENAI_API_KEY")

    # 3. データセットの準備
    trainset = [
        dspy.Example(
            inquiry="クレジットカードが不正利用された可能性があります！即時停止してください！",
            expected_priority="High",
            expected_category="Billing",
        ).with_inputs("inquiry"),
        dspy.Example(
            inquiry="ログインパスワードを忘れてしまいログインできません。",
            expected_priority="Medium",
            expected_category="Account",
        ).with_inputs("inquiry"),
        dspy.Example(
            inquiry="APIのレスポンスが500エラーを返してシステム連携が停止しています。",
            expected_priority="High",
            expected_category="Technical",
        ).with_inputs("inquiry"),
        dspy.Example(
            inquiry="領収書の宛名を変更して再発行をお願いできますでしょうか。",
            expected_priority="Low",
            expected_category="Billing",
        ).with_inputs("inquiry"),
        dspy.Example(
            inquiry="貴社のサービス導入を検討中ですが、パンフレットを送付いただけますか？",
            expected_priority="Low",
            expected_category="General",
        ).with_inputs("inquiry"),
    ]

    devset = [
        dspy.Example(
            inquiry="本番サーバーでデータベース接続エラーが頻発しています。",
            expected_priority="High",
            expected_category="Technical",
        ).with_inputs("inquiry"),
        dspy.Example(
            inquiry="アカウントの登録メールアドレスを変更する方法を教えてください。",
            expected_priority="Low",
            expected_category="Account",
        ).with_inputs("inquiry"),
    ]

    print("=== MIPROv2 Instruction 自動最適化 ===")
    print(f"訓練データ: {len(trainset)}件, 検証データ: {len(devset)}件\n")

    if not api_key:
        print("[Notice] OPENAI_API_KEY が未設定です。")
        print("MIPROv2 では Meta-LLM が Signature とデータセットを解析し、")
        print("最適なプロンプト（指示文 + デモ例）を自動で探索・生成します。\n")
        print("--- MIPROv2 の実行コード例 ---")
        print("""
    # 1. LMの初期化
    task_lm = dspy.LM("openai/gpt-4o-mini")
    prompt_lm = dspy.LM("openai/gpt-4o-mini")  # 指示文生成用のMeta-LLM
    dspy.configure(lm=task_lm)

    # 2. MIPROv2 Optimizerの初期化
    optimizer = MIPROv2(
        metric=triage_metric,
        prompt_model=prompt_lm,
        task_model=task_lm,
        num_candidates=5,      # 生成する指示文候補の数
        num_trials=10,         # ベイズ探索（Optuna）の試行回数
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
        auto="light",          # 'light', 'medium', 'heavy' から選択可能
    )

    # 3. コンパイル（指示文とデモの自動最適化）
    compiled_module = optimizer.compile(
        student=SupportTriageModule(),
        trainset=trainset,
        valset=devset,
    )

    # 4. 最適化前後のプロンプト比較
    print("--- 最適化後のモジュール ---")
    compiled_module.save("outputs/support_triage_mipro.json")
        """)
        return

    # APIキーがある場合の実際の最適化実行
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    base_module = SupportTriageModule()
    evaluator = dspy.evaluate.Evaluate(devset=devset, metric=triage_metric, display_table=False)

    print("--- ベースライン評価 ---")
    base_score = evaluator(base_module)
    print(f"Base Score: {base_score:.2f}%")

    print("\n--- MIPROv2 最適化開始 (auto='light') ---")
    teleprompter = MIPROv2(
        metric=triage_metric,
        auto="light",
        num_candidates=3,
        num_trials=5,
    )
    compiled_module = teleprompter.compile(
        student=base_module,
        trainset=trainset,
        valset=devset,
    )

    print("\n--- 最適化後評価 ---")
    opt_score = evaluator(compiled_module)
    print(f"Optimized Score: {opt_score:.2f}%")

    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/support_triage_mipro.json"
    compiled_module.save(save_path)
    print(f"最適化済みモジュールを保存しました: {save_path}")


if __name__ == "__main__":
    main()
