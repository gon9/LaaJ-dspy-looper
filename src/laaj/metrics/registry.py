"""メトリクスレジストリ"""

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dspy

from laaj.metrics.exact_match import exact_match_metric
from laaj.metrics.llm_judge import create_llm_judge_metric
from laaj.metrics.semantic_similarity import create_semantic_similarity_metric


class MetricRegistry:
    """メトリクス関数を管理するレジストリ"""

    _builtin_metrics: dict[
        str, Callable[..., Callable[[dspy.Example, dspy.Prediction, Any], float]]
    ] = {
        "exact_match": exact_match_metric,
        "llm_judge": create_llm_judge_metric,
        "semantic_similarity": create_semantic_similarity_metric,
    }

    @classmethod
    def get_metric(
        cls,
        name: str,
        field: str | None = None,
        custom_path: str | None = None,
        custom_func: str | None = None,
        **kwargs: Any,
    ) -> Callable[[dspy.Example, dspy.Prediction, Any], float]:
        """メトリクス関数を取得

        Args:
            name: メトリクス名（組み込みメトリクスの場合）
            field: 比較対象のフィールド名
            custom_path: カスタムメトリクスのファイルパス
            custom_func: カスタムメトリクスの関数名
            **kwargs: メトリクスファクトリに渡す追加引数

        Returns:
            callable: メトリクス関数

        Raises:
            ValueError: メトリクスが見つからない場合、または必須引数が不足している場合
        """
        if custom_path and custom_func:
            return cls._load_custom_metric(custom_path, custom_func)

        if name == "exact_match":
            if field is None:
                raise ValueError("組み込みメトリクス 'exact_match' には field パラメータが必要です")
            return exact_match_metric(field)

        elif name == "llm_judge":
            criterion = kwargs.get(
                "criterion", "期待される回答の内容・意図を満たしているかを判定してください。"
            )
            threshold = kwargs.get("threshold", 0.7)
            field_name = field or "label"
            return create_llm_judge_metric(
                criterion=criterion,
                threshold=threshold,
                field_gold=field_name,
                field_pred=field_name,
            )

        elif name == "semantic_similarity":
            threshold = kwargs.get("threshold", 0.6)
            field_name = field or "label"
            return create_semantic_similarity_metric(
                field_gold=field_name,
                field_pred=field_name,
                threshold=threshold,
            )

        elif name in cls._builtin_metrics:
            factory = cls._builtin_metrics[name]
            if field is not None:
                return factory(field, **kwargs)
            return factory(**kwargs)

        raise ValueError(f"メトリクス '{name}' が見つかりません")

    @classmethod
    def _load_custom_metric(cls, path: str, func_name: str) -> Callable:
        """カスタムメトリクスをロード

        Args:
            path: Pythonファイルのパス
            func_name: 関数名

        Returns:
            callable: メトリクス関数

        Raises:
            FileNotFoundError: ファイルが存在しない場合
            AttributeError: 関数が見つからない場合
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"カスタムメトリクスファイルが見つかりません: {path}")

        spec = importlib.util.spec_from_file_location("custom_metric", file_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"モジュールのロードに失敗しました: {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["custom_metric"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, func_name):
            raise AttributeError(f"関数 '{func_name}' が見つかりません: {path}")

        return getattr(module, func_name)
