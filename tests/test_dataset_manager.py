"""Dataset Managerのテスト"""

import json
import tempfile
from pathlib import Path

import pytest

from laaj.dataset import DatasetManager


def test_load_valid_dataset():
    """正常なデータセットの読み込みテスト"""
    data = [
        {"text": "sample1", "label": "A"},
        {"text": "sample2", "label": "B"},
        {"text": "sample3", "label": "A"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        temp_path = f.name

    try:
        manager = DatasetManager(temp_path, input_fields=["text"])
        manager.load()
        assert manager.size == 3
    finally:
        Path(temp_path).unlink()


def test_split_dataset():
    """データセット分割のテスト"""
    data = [{"text": f"sample{i}", "label": "A"} for i in range(100)]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        temp_path = f.name

    try:
        manager = DatasetManager(temp_path, input_fields=["text"])
        manager.load()
        train, val, test = manager.split(0.7, 0.15, 0.15)

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
    finally:
        Path(temp_path).unlink()


def test_missing_dataset_file():
    """存在しないデータセットファイルのテスト"""
    manager = DatasetManager("nonexistent.jsonl", input_fields=["text"])
    with pytest.raises(FileNotFoundError):
        manager.load()


def test_invalid_jsonl():
    """不正なJSONL形式のテスト"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"text": "valid"}\n')
        f.write('invalid json line\n')
        temp_path = f.name

    try:
        manager = DatasetManager(temp_path, input_fields=["text"])
        with pytest.raises(ValueError, match="JSONL形式エラー"):
            manager.load()
    finally:
        Path(temp_path).unlink()
