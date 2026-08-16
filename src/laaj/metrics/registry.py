"""メトリクスレジストリ"""

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import dspy

from laaj.metrics.exact_match import exact_match_metric


class MetricRegistry:
    """メトリクス関数を管理するレジストリ"""

    _builtin_metrics = {
        "exact_match": exact_match_metric,
    }

    @classmethod
    def get_metric(
        cls,
        name: str,
        field: str | None = None,
        custom_path: str | None = None,
        custom_func: str | None = None,
    ) -> Callable[[dspy.Example, dspy.Prediction, any], float]:
        """メトリクス関数を取得

        Args:
            name: メトリクス名（組み込みメトリクスの場合）
            field: 比較対象のフィールド名
            custom_path: カスタムメトリクスのファイルパス
            custom_func: カスタムメトリクスの関数名

        Returns:
            callable: メトリクス関数

        Raises:
            ValueError: メトリクスが見つからない場合
        """
        if custom_path and custom_func:
            return cls._load_custom_metric(custom_path, custom_func)

        if name in cls._builtin_metrics:
            metric_factory = cls._builtin_metrics[name]
            if field is None:
                raise ValueError(f"組み込みメトリクス '{name}' には field パラメータが必要です")
            return metric_factory(field)

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
