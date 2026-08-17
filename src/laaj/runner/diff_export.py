"""差分比較およびエクスポート Runner"""

from pathlib import Path
from typing import Any

from laaj.config import Config, load_config
from laaj.export.exporter import ModuleInspector
from laaj.modules.registry import ModuleRegistry


class DiffRunner:
    """最適化前後のプロンプト差分を抽出する Runner"""

    def __init__(self, config: Config):
        self.config = config

    @classmethod
    def from_config_path(cls, config_path: str | Path) -> "DiffRunner":
        cfg = load_config(config_path)
        return cls(cfg)

    def run(self, module_path: str | Path) -> dict[str, Any]:
        """差分データを取得"""
        cfg = self.config
        base_module = ModuleRegistry.instantiate_module(cfg.module.path, cfg.module.class_name)
        opt_module = ModuleRegistry.instantiate_module(cfg.module.path, cfg.module.class_name)
        opt_module.load(module_path)

        return ModuleInspector.diff_modules(base_module, opt_module)


class ExportRunner:
    """最適化済みモジュールをエクスポートする Runner"""

    def __init__(self, config: Config):
        self.config = config

    @classmethod
    def from_config_path(cls, config_path: str | Path) -> "ExportRunner":
        cfg = load_config(config_path)
        return cls(cfg)

    def run(
        self,
        module_path: str | Path,
        output_format: str = "json",
        output_file: str | Path | None = None,
    ) -> str:
        """エクスポートを実行"""
        cfg = self.config
        content = ModuleInspector.export_module(
            module_path=cfg.module.path,
            class_name=cfg.module.class_name,
            saved_path=str(module_path),
            output_format=output_format,
        )

        if output_file:
            Path(output_file).write_text(content, encoding="utf-8")

        return content
