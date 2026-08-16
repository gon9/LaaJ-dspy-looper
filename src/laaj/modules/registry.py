"""DSPy Moduleレジストリ"""

import importlib.util
import sys
from pathlib import Path

import dspy


class ModuleRegistry:
    """DSPy Moduleを管理するレジストリ"""

    @classmethod
    def load_module(cls, module_path: str, class_name: str) -> type[dspy.Module]:
        """ユーザ定義のDSPy Moduleをロード

        Args:
            module_path: Pythonファイルのパス
            class_name: クラス名

        Returns:
            type[dspy.Module]: DSPy Moduleクラス

        Raises:
            FileNotFoundError: ファイルが存在しない場合
            AttributeError: クラスが見つからない場合
            TypeError: DSPy Moduleでない場合
        """
        file_path = Path(module_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Moduleファイルが見つかりません: {module_path}")

        spec = importlib.util.spec_from_file_location("user_module", file_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"モジュールのロードに失敗しました: {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["user_module"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, class_name):
            raise AttributeError(f"クラス '{class_name}' が見つかりません: {module_path}")

        module_class = getattr(module, class_name)

        if not issubclass(module_class, dspy.Module):
            raise TypeError(f"クラス '{class_name}' はdspy.Moduleを継承していません")

        return module_class

    @classmethod
    def instantiate_module(cls, module_path: str, class_name: str) -> dspy.Module:
        """DSPy Moduleをインスタンス化

        Args:
            module_path: Pythonファイルのパス
            class_name: クラス名

        Returns:
            dspy.Module: DSPy Moduleインスタンス
        """
        module_class = cls.load_module(module_path, class_name)
        return module_class()
