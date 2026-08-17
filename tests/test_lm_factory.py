"""LM Factoryのテスト"""

from unittest.mock import patch

import pytest

from laaj.config import LMConfig
from laaj.exceptions import ConfigError
from laaj.lm.factory import LMFactory


def test_create_lm_success():
    """正常な設定でLMが生成されることをテスト"""
    config = LMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="sk-test-key",
        temperature=0.7,
        max_tokens=2000,
    )
    lm = LMFactory.create_lm(config)
    assert lm is not None
    assert lm.model == "openai/gpt-4o-mini"
    assert lm.kwargs.get("temperature") == 0.7


def test_create_lm_missing_api_key(monkeypatch):
    """APIキーが未設定の場合にConfigErrorが発生することをテスト"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = LMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key=None,
    )
    with pytest.raises(ConfigError, match="LLM APIキーが設定されていません"):
        LMFactory.create_lm(config)


@patch("dspy.configure")
def test_configure_default_lm(mock_configure):
    """グローバルLMが設定されることをテスト"""
    config = LMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="sk-test-key",
    )
    lm = LMFactory.configure_default_lm(config)
    mock_configure.assert_called_once_with(lm=lm)
