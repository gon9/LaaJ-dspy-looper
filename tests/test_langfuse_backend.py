"""Langfuse Observability バックエンドのテスト"""

from unittest.mock import patch

from laaj.config import ObservabilityConfig
from laaj.observability.langfuse_backend import LangfuseBackend


def test_langfuse_unconfigured():
    """キーが未設定の場合に初期化がFalseを返すことをテスト"""
    config = ObservabilityConfig(
        backend="langfuse",
        langfuse_public_key=None,
        langfuse_secret_key=None,
    )
    backend = LangfuseBackend(config)
    success = backend.initialize()
    assert success is False
    assert backend.is_initialized is False


@patch("openinference.instrumentation.dspy.DSPyInstrumentor.instrument")
@patch("opentelemetry.trace.set_tracer_provider")
def test_langfuse_configured(mock_set_tracer, mock_instrument):
    """キーが設定されている場合に初期化が成功することをテスト"""
    config = ObservabilityConfig(
        backend="langfuse",
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
        langfuse_base_url="https://cloud.langfuse.com",
    )
    backend = LangfuseBackend(config)
    success = backend.initialize()
    assert success is True
    assert backend.is_initialized is True
    mock_instrument.assert_called_once()


def test_langfuse_log_experiment_metadata():
    """実験メタデータの記録メソッド呼び出しをテスト"""
    config = ObservabilityConfig(
        backend="langfuse",
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
    )
    backend = LangfuseBackend(config)
    # 例外が起きずに実行できることを確認
    backend.log_experiment_metadata(
        experiment_name="test_exp",
        config={"key": "val"},
        results={"score": 0.95},
    )
