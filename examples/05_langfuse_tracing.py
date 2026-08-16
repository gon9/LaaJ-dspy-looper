"""05_langfuse_tracing.py - Langfuse による DSPy トレーシング連携

本スクリプトでは以下を学びます：
1. openinference-instrumentation-dspy を使った DSPy の自動計装 (Instrumentation)
2. Langfuse へのトレース送信設定
3. 推論時および最適化時の LLM 呼び出し履歴の可視化
"""

import os

import dspy


class TextSummarizer(dspy.Signature):
    """長い記事テキストを箇条書き3点で要約する。"""

    article_text: str = dspy.InputField(desc="元記事本文")
    summary: str = dspy.OutputField(desc="箇条書き3点の要約")


def setup_langfuse_tracing():
    """Langfuse トレーシングを初期化する。"""
    try:
        from openinference.instrumentation.dspy import DSPyInstrumentor
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Langfuse用のエンドポイントと認証情報
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not public_key or not secret_key:
            print("[Info] LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY が未設定です。")
            print(
                "Langfuse クラウドまたはセルフホスト環境のキーを設定するとトレースが送信されます。\n"
            )
            return False

        # OpenTelemetry の設定
        import base64

        auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        headers = {"Authorization": f"Basic {auth}"}

        exporter = OTLPSpanExporter(
            endpoint=f"{host.rstrip('/')}/api/public/otel/v1/traces",
            headers=headers,
        )

        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(tracer_provider)

        # DSPy の自動計装
        DSPyInstrumentor().instrument()
        print("✓ Langfuse トレーシングを初期化しました。")
        return True

    except ImportError:
        print(
            "[Warning] openinference-instrumentation-dspy または opentelemetry がインストールされていません。"
        )
        print("インストール: uv pip install -e '.[observability]'")
        return False


def main():
    print("=== DSPy & Langfuse トレーシング ===")

    is_traced = setup_langfuse_tracing()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[Notice] OPENAI_API_KEY が未設定です。")
        print("Langfuse 連携時のコード構造:")
        print("""
    # 1. トレーシングの計装
    setup_langfuse_tracing()

    # 2. LMの設定
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    # 3. DSPyモジュールの実行
    summarizer = dspy.ChainOfThought(TextSummarizer)
    result = summarizer(article_text="...")

    # -> Langfuse ダッシュボード上に以下が自動記録されます：
    #    - プロンプトテンプレートと入力値
    #    - LLMの推論トークン数・レイテンシ・コスト
    #    - ChainOfThought の中間推論ステップ
        """)
        return

    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    summarizer = dspy.ChainOfThought(TextSummarizer)
    sample_article = """
    DSPyはスタンフォード大学発の宣言的LLMプログラミングフレームワークです。
    手作業でのプロンプト調整に代わり、入出力の仕様（Signature）と処理ロジック（Module）をコードで記述し、
    評価関数（Metric）に基づいて自動的にプロンプトやFew-Shot例を最適化（Compile）します。
    これにより、LLMのモデル変更時にもプロンプトを手動で書き直す必要がなくなります。
    """

    print("要約モジュールを実行中...")
    result = summarizer(article_text=sample_article)
    print("\n--- 要約結果 ---")
    print(result.summary)

    if is_traced:
        print("\n✓ Langfuse にトレースが送信されました。Langfuse UI で確認してください。")


if __name__ == "__main__":
    main()
