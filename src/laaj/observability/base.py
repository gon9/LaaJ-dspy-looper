"""Observability抽象基底クラス"""

from abc import ABC, abstractmethod
from typing import Any


class BaseObservability(ABC):
    """Observabilityバックエンドの共通インターフェース"""

    @abstractmethod
    def initialize(self) -> bool:
        """バックエンドの初期化（計装、接続など）

        Returns:
            bool: 初期化に成功したかどうか
        """
        pass

    @abstractmethod
    def log_experiment(
        self,
        experiment_name: str,
        config: dict[str, Any],
        results: dict[str, Any],
        optimized_module_path: str,
    ) -> str | None:
        """実験結果を記録

        Args:
            experiment_name: 実験名
            config: 設定情報
            results: 実験結果（スコア等）
            optimized_module_path: 最適化済みモジュールの保存パス

        Returns:
            str | None: ログファイルパスまたは識別子
        """
        pass

    @abstractmethod
    def log_evaluation(
        self,
        experiment_name: str,
        module_type: str,
        scores: dict[str, float],
        module_path: str | None = None,
    ) -> str | None:
        """評価結果を記録

        Args:
            experiment_name: 実験名
            module_type: モジュール種別 ("baseline" または "optimized")
            scores: スコア辞書
            module_path: 評価したモジュールのパス

        Returns:
            str | None: ログファイルパスまたは識別子
        """
        pass
