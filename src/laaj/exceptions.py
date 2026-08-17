"""LaaJ-dspy-looper ドメイン例外定義"""


class LaaJError(Exception):
    """LaaJ-dspy-looper 基底例外クラス"""

    pass


class ConfigError(LaaJError):
    """設定ファイルの読み込み・構文・バリデーションエラー"""

    pass


class DatasetError(LaaJError):
    """データセットの読み込み・分割・フォーマットエラー"""

    pass


class ModuleLoadError(LaaJError):
    """DSPy Moduleのロード・インスタンス化エラー"""

    pass


class MetricError(LaaJError):
    """評価メトリクスの解決・実行エラー"""

    pass


class OptimizationError(LaaJError):
    """Optimizerの生成・コンパイル実行エラー"""

    pass


class ObservabilityError(LaaJError):
    """Observabilityバックエンドのエラー"""

    pass
