"""設定ファイルの読み込みとバリデーション"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class DatasetSplitConfig(BaseModel):
    """データセット分割設定"""

    train: float = Field(ge=0.0, le=1.0)
    val: float = Field(ge=0.0, le=1.0)
    test: float = Field(ge=0.0, le=1.0)

    @field_validator("train", "val", "test")
    @classmethod
    def validate_split(cls, v: float) -> float:
        """分割比率のバリデーション"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("分割比率は0.0から1.0の間である必要があります")
        return v

    def model_post_init(self, __context: Any) -> None:
        """分割比率の合計が1.0であることを確認"""
        total = self.train + self.val + self.test
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"分割比率の合計は1.0である必要があります（現在: {total}）")


class DatasetConfig(BaseModel):
    """データセット設定"""

    path: str
    input_fields: list[str]
    split: DatasetSplitConfig


class ModuleConfig(BaseModel):
    """DSPy Module設定"""

    path: str
    class_name: str


class LMConfig(BaseModel):
    """言語モデル設定"""

    provider: str
    model: str
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=16000, gt=0)
    api_key: str | None = None
    api_base: str | None = None


class OptimizerConfig(BaseModel):
    """Optimizer設定"""

    name: Literal[
        "MIPROv2",
        "BootstrapFewShot",
        "BootstrapFewShotWithRandomSearch",
        "KNNFewShot",
        "COPRO",
    ]
    params: dict[str, Any] = Field(default_factory=dict)


class MetricConfig(BaseModel):
    """評価メトリクス設定"""

    name: str
    field: str | None = None
    custom_path: str | None = None
    custom_func: str | None = None


class ObservabilityConfig(BaseModel):
    """Observability設定"""

    backend: Literal["local", "langfuse"] = "local"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str | None = None


class ExperimentConfig(BaseModel):
    """実験全体の設定"""

    name: str
    description: str = ""


class Config(BaseModel):
    """全体設定"""

    experiment: ExperimentConfig
    dataset: DatasetConfig
    module: ModuleConfig
    lm: LMConfig
    optimizer: OptimizerConfig
    metric: MetricConfig
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)


def load_config(config_path: str | Path) -> Config:
    """YAML設定ファイルを読み込む

    Args:
        config_path: 設定ファイルのパス

    Returns:
        Config: パース済み設定オブジェクト

    Raises:
        FileNotFoundError: 設定ファイルが存在しない場合
        ValueError: YAML形式が不正な場合
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    if not config_dict:
        raise ValueError("設定ファイルが空です")

    return Config(**config_dict)
