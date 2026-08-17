"""Langfuse Observability バックエンド"""

import base64
import logging
from typing import Any

from laaj.config import ObservabilityConfig
from laaj.observability.base import BaseObservability

logger = logging.getLogger(__name__)


class LangfuseBackend(BaseObservability):
    """Langfuse / OpenTelemetry によるトレーシング・実験管理バックエンド"""

    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self._is_initialized = False

    def initialize(self) -> bool:
        """Langfuse OTLP エクスポータと DSPy 計装を初期化

        Returns:
            bool: 初期化が成功したかどうか
        """
        public_key = self.config.langfuse_public_key
        secret_key = self.config.langfuse_secret_key
        host = self.config.langfuse_base_url or "https://cloud.langfuse.com"

        if not public_key or not secret_key:
            logger.warning("Langfuse APIキー（public_key / secret_key）が設定されていません。")
            return False

        try:
            from openinference.instrumentation.dspy import DSPyInstrumentor
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
            headers = {"Authorization": f"Basic {auth}"}

            endpoint = f"{host.rstrip('/')}/api/public/otel/v1/traces"
            exporter = OTLPSpanExporter(
                endpoint=endpoint,
                headers=headers,
            )

            tracer_provider = TracerProvider()
            tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(tracer_provider)

            DSPyInstrumentor().instrument()
            self._is_initialized = True
            return True

        except Exception as e:
            logger.error(f"Langfuse の初期化に失敗しました: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def log_experiment(
        self,
        experiment_name: str,
        config: dict[str, Any],
        results: dict[str, Any],
        optimized_module_path: str,
    ) -> str | None:
        """実験結果を記録"""
        logger.info(
            f"Langfuse 実験メタデータを記録: {experiment_name}, results={results}, path={optimized_module_path}"
        )
        return f"langfuse://experiments/{experiment_name}"

    def log_evaluation(
        self,
        experiment_name: str,
        module_type: str,
        scores: dict[str, float],
        module_path: str | None = None,
    ) -> str | None:
        """評価結果を記録"""
        logger.info(
            f"Langfuse 評価メタデータを記録: {experiment_name}, module_type={module_type}, scores={scores}"
        )
        return f"langfuse://evaluations/{experiment_name}_{module_type}"

    def log_experiment_metadata(
        self,
        experiment_name: str,
        config: dict[str, Any],
        results: dict[str, Any],
    ) -> None:
        """実験メタデータを記録 (互換性用)"""
        logger.info(f"Langfuse 実験メタデータを記録: {experiment_name}, results={results}")
