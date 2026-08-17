"""言語モデル (dspy.LM) 管理ファクトリ"""

import os

import dspy

from laaj.config import LMConfig
from laaj.exceptions import ConfigError


class LMFactory:
    """dspy.LM インスタンスの生成・設定を一元管理するファクトリ"""

    @staticmethod
    def create_lm(config: LMConfig) -> dspy.LM:
        """LMConfigからdspy.LMインスタンスを生成

        Args:
            config: 言語モデル設定

        Returns:
            dspy.LM: 初期化済みLMインスタンス

        Raises:
            ConfigError: APIキーが設定されていない場合
        """
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigError(
                f"LLM APIキーが設定されていません (provider={config.provider}, model={config.model})。"
                "環境変数 OPENAI_API_KEY または設定ファイル内の lm.api_key を指定してください。"
            )

        model_str = f"{config.provider}/{config.model}"
        return dspy.LM(
            model_str,
            api_key=api_key,
            api_base=config.api_base,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    @classmethod
    def configure_default_lm(cls, config: LMConfig) -> dspy.LM:
        """DSPyのグローバルLMとして設定

        Args:
            config: 言語モデル設定

        Returns:
            dspy.LM: 設定されたLMインスタンス
        """
        lm = cls.create_lm(config)
        dspy.configure(lm=lm)
        return lm
