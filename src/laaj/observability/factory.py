"""Observabilityファクトリ"""

from pathlib import Path

from laaj.config import ObservabilityConfig
from laaj.observability.base import BaseObservability
from laaj.observability.langfuse_backend import LangfuseBackend
from laaj.observability.local_backend import LocalLogger


class ObservabilityFactory:
    """Observabilityバックエンドを生成するファクトリ"""

    @staticmethod
    def create_backend(
        config: ObservabilityConfig,
        output_dir: str | Path = "outputs",
    ) -> BaseObservability:
        """設定に基づいて適切なObservabilityバックエンドを生成

        Args:
            config: Observability設定
            output_dir: ローカルログ出力ディレクトリ

        Returns:
            BaseObservability: 初期化可能なバックエンドインスタンス
        """
        if config.backend == "langfuse":
            backend = LangfuseBackend(config)
            if backend.initialize():
                return backend
            # 初期化失敗時はLocalLoggerにフォールバック
            return LocalLogger(output_dir=output_dir)

        return LocalLogger(output_dir=output_dir)
