"""設定モジュールのテスト"""

import tempfile
from pathlib import Path

import pytest
import yaml

from laaj.config import Config, load_config


def test_load_valid_config():
    """正常な設定ファイルの読み込みテスト"""
    config_data = {
        "experiment": {"name": "test_exp", "description": "test description"},
        "dataset": {
            "path": "data/test.jsonl",
            "input_fields": ["text"],
            "split": {"train": 0.7, "val": 0.15, "test": 0.15},
        },
        "module": {"path": "modules/test.py", "class_name": "TestModule"},
        "lm": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 1.0},
        "optimizer": {"name": "MIPROv2", "params": {"num_candidates": 5}},
        "metric": {"name": "exact_match", "field": "label"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        config = load_config(temp_path)
        assert config.experiment.name == "test_exp"
        assert config.dataset.path == "data/test.jsonl"
        assert config.optimizer.name == "MIPROv2"
    finally:
        Path(temp_path).unlink()


def test_invalid_split_ratio():
    """不正な分割比率のテスト"""
    with pytest.raises(ValueError, match="分割比率の合計は1.0である必要があります"):
        Config(
            experiment={"name": "test", "description": ""},
            dataset={
                "path": "data/test.jsonl",
                "input_fields": ["text"],
                "split": {"train": 0.5, "val": 0.3, "test": 0.3},
            },
            module={"path": "modules/test.py", "class_name": "TestModule"},
            lm={"provider": "openai", "model": "gpt-4o-mini"},
            optimizer={"name": "MIPROv2"},
            metric={"name": "exact_match", "field": "label"},
        )


def test_missing_config_file():
    """存在しない設定ファイルのテスト"""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")
