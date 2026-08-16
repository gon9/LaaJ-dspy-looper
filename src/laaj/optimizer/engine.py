"""Optimizer実行エンジン"""

from typing import Any, Callable

import dspy

from laaj.config import OptimizerConfig


class OptimizerEngine:
    """DSPy Optimizerを実行するエンジン"""

    @staticmethod
    def create_optimizer(
        config: OptimizerConfig,
        metric: Callable[[dspy.Example, dspy.Prediction, Any], float],
    ) -> Any:
        """設定からOptimizerインスタンスを作成

        Args:
            config: Optimizer設定
            metric: 評価メトリクス関数

        Returns:
            DSPy Optimizerインスタンス

        Raises:
            ValueError: サポートされていないOptimizer名の場合
        """
        optimizer_name = config.name
        params = config.params.copy()

        params["metric"] = metric

        if optimizer_name == "MIPROv2":
            return dspy.MIPROv2(**params)
        elif optimizer_name == "BootstrapFewShot":
            return dspy.BootstrapFewShot(**params)
        elif optimizer_name == "BootstrapFewShotWithRandomSearch":
            return dspy.BootstrapFewShotWithRandomSearch(**params)
        elif optimizer_name == "KNNFewShot":
            return dspy.KNNFewShot(**params)
        elif optimizer_name == "COPRO":
            return dspy.COPRO(**params)
        else:
            raise ValueError(f"サポートされていないOptimizer: {optimizer_name}")

    @staticmethod
    def optimize(
        optimizer: Any,
        student: dspy.Module,
        trainset: list[dspy.Example],
        valset: list[dspy.Example] | None = None,
    ) -> dspy.Module:
        """Optimizerを実行してモジュールを最適化

        Args:
            optimizer: DSPy Optimizerインスタンス
            student: 最適化対象のDSPy Module
            trainset: 訓練データセット
            valset: 検証データセット（オプション）

        Returns:
            dspy.Module: 最適化済みモジュール
        """
        if valset is not None:
            optimized_module = optimizer.compile(
                student=student, trainset=trainset, valset=valset
            )
        else:
            optimized_module = optimizer.compile(student=student, trainset=trainset)

        return optimized_module
