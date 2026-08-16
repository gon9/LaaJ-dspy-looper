"""最適化モジュールのエクスポート & 差分比較"""

import json
from pathlib import Path
from typing import Any

import dspy

from laaj.modules.registry import ModuleRegistry


class ModuleInspector:
    """DSPyモジュールの内部情報（Instruction, Demos等）を解析・抽出するクラス"""

    @classmethod
    def extract_info(cls, module: dspy.Module) -> dict[str, Any]:
        """モジュールの内部構成情報を抽出"""
        info = {
            "predictors": [],
        }

        for name, pred in module.named_predictors():
            sig = getattr(pred, "signature", None)
            demos = getattr(pred, "demos", [])

            pred_info = {
                "name": name,
                "instructions": getattr(sig, "instructions", "") if sig else "",
                "demos_count": len(demos),
                "demos": [
                    {k: v for k, v in d.items() if not k.startswith("_")}
                    for d in demos
                    if isinstance(d, dict) or hasattr(d, "items")
                ],
            }
            info["predictors"].append(pred_info)

        return info

    @classmethod
    def diff_modules(
        cls,
        base_module: dspy.Module,
        opt_module: dspy.Module,
    ) -> dict[str, Any]:
        """2つのモジュールの差分情報を抽出"""
        base_info = cls.extract_info(base_module)
        opt_info = cls.extract_info(opt_module)

        diffs = []
        base_preds = {p["name"]: p for p in base_info["predictors"]}
        opt_preds = {p["name"]: p for p in opt_info["predictors"]}

        for name, opt_p in opt_preds.items():
            base_p = base_preds.get(name, {"instructions": "", "demos_count": 0, "demos": []})
            inst_changed = base_p["instructions"] != opt_p["instructions"]
            demos_added = opt_p["demos_count"] - base_p["demos_count"]

            diffs.append(
                {
                    "predictor_name": name,
                    "instruction_changed": inst_changed,
                    "base_instruction": base_p["instructions"],
                    "optimized_instruction": opt_p["instructions"],
                    "base_demos_count": base_p["demos_count"],
                    "optimized_demos_count": opt_p["demos_count"],
                    "demos_added": demos_added,
                    "new_demos": opt_p["demos"],
                }
            )

        return {
            "diffs": diffs,
        }

    @classmethod
    def export_module(
        cls,
        module_path: str,
        class_name: str,
        saved_path: str,
        output_format: str = "json",
    ) -> str:
        """最適化済みモジュールを指定フォーマットでエクスポート"""
        module = ModuleRegistry.instantiate_module(module_path, class_name)
        if Path(saved_path).exists():
            module.load(saved_path)

        info = cls.extract_info(module)

        if output_format == "json":
            return json.dumps(info, ensure_ascii=False, indent=2)
        elif output_format == "markdown":
            lines = ["# Optimized DSPy Module Export\n"]
            for pred in info["predictors"]:
                lines.append(f"## Predictor: `{pred['name']}`\n")
                lines.append(f"**Instructions**:\n```\n{pred['instructions']}\n```\n")
                lines.append(f"**Few-Shot Demos ({pred['demos_count']} items)**:\n")
                for i, demo in enumerate(pred["demos"], 1):
                    lines.append(
                        f"### Demo {i}\n```json\n{json.dumps(demo, ensure_ascii=False, indent=2)}\n```\n"
                    )
            return "\n".join(lines)
        else:
            raise ValueError(f"サポートされていないフォーマット: {output_format}")
