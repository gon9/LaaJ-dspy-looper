"""Module Registryのテスト"""

import tempfile
from pathlib import Path

import dspy
import pytest

from laaj.modules.registry import ModuleRegistry


def test_load_valid_module():
    """正常なModuleのロードをテスト"""
    module_class = ModuleRegistry.load_module(
        module_path="examples/simple_classifier.py",
        class_name="SimpleClassifier",
    )
    assert issubclass(module_class, dspy.Module)


def test_instantiate_valid_module():
    """正常なModuleのインスタンス化をテスト"""
    module = ModuleRegistry.instantiate_module(
        module_path="examples/simple_classifier.py",
        class_name="SimpleClassifier",
    )
    assert isinstance(module, dspy.Module)


def test_load_nonexistent_file():
    """存在しないファイルのロードでエラーが発生することをテスト"""
    with pytest.raises(FileNotFoundError):
        ModuleRegistry.load_module(
            module_path="nonexistent/path.py",
            class_name="DummyClass",
        )


def test_load_nonexistent_class():
    """存在しないクラスのロードでエラーが発生することをテスト"""
    with pytest.raises(AttributeError, match="クラス 'NonExistentClass' が見つかりません"):
        ModuleRegistry.load_module(
            module_path="examples/simple_classifier.py",
            class_name="NonExistentClass",
        )


def test_load_non_dspy_module():
    """dspy.Moduleを継承していないクラスのロードでエラーが発生することをテスト"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "invalid_module.py"
        test_file.write_text("class NotADSPyModule:\n    pass\n")

        with pytest.raises(TypeError, match="dspy.Moduleを継承していません"):
            ModuleRegistry.load_module(
                module_path=str(test_file),
                class_name="NotADSPyModule",
            )
