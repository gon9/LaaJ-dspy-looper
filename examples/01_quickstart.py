"""01_quickstart.py - DSPyの基本概念と最小限の実行

本スクリプトでは以下を学びます：
1. LM（言語モデル）の設定
2. Signature（入出力仕様）の定義（インライン記法 & クラス記法）
3. Module（dspy.Predict と dspy.ChainOfThought）の実行
4. dspy.inspect_history() によるプロンプトの可視化
"""

import os

import dspy


# 1. Signatureの定義
# ----------------------------------------------------
# クラスベースのSignature定義（推奨：型や説明文を明記できる）
class QAWithContext(dspy.Signature):
    """提供されたコンテキストに基づいて質問に簡潔に回答するタスク。"""

    context: str = dspy.InputField(desc="背景情報・参考文書")
    question: str = dspy.InputField(desc="質問内容")
    answer: str = dspy.OutputField(desc="1〜2文程度の簡潔な回答")


def main():
    # 2. LMの設定
    # 環境変数 OPENAI_API_KEY が設定されている場合は OpenAI、
    # 未設定の場合はローカルLLMやダミー設定の案内を表示
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "[Notice] OPENAI_API_KEY が設定されていないため、DSPyのモジュール定義のみを実演します。"
        )
        print('実際にLLMを呼び出すには: export OPENAI_API_KEY="sk-..." を設定してください。\n')
        lm = dspy.LM("openai/gpt-4o-mini", api_key="dummy_key_for_inspection")
        dspy.configure(lm=lm)
    else:
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm)

    # 3. Moduleの構築 (dspy.Predict vs dspy.ChainOfThought)
    print("=== 1. dspy.Predict（直接回答） ===")
    predict_module = dspy.Predict(QAWithContext)
    print(f"Predict Module: {predict_module}")

    print("\n=== 2. dspy.ChainOfThought（思考プロセス付き回答） ===")
    cot_module = dspy.ChainOfThought(QAWithContext)
    print(f"ChainOfThought Module: {cot_module}")

    # インラインSignatureの例
    inline_module = dspy.Predict("sentence -> sentiment: str")
    print(f"\n=== 3. Inline Signature Module ===\n{inline_module}")

    # 4. 実行例（APIキーがある場合）
    sample_context = (
        "LaaJ-dspy-looperは、DSPyとLLM-as-a-Judgeを組み合わせたプロンプト自動最適化ツールです。"
    )
    sample_question = "LaaJ-dspy-looperは何を行うツールですか？"

    if api_key:
        print("\n=== 推論の実行 (ChainOfThought) ===")
        response = cot_module(context=sample_context, question=sample_question)
        print(f"Reasoning (思考理由): {response.reasoning}")
        print(f"Answer (回答): {response.answer}")

        print("\n=== 生成プロンプトの確認 (inspect_history) ===")
        dspy.inspect_history(n=1)
    else:
        print("\n[TIP] LM呼び出し時の入力例:")
        print(f"  context: {sample_context}")
        print(f"  question: {sample_question}")
        print(
            "  dspy.ChainOfThought を使うと、LLMは自動的に 'reasoning' (思考過程) を出力した上で 'answer' を導きます。"
        )


if __name__ == "__main__":
    main()
