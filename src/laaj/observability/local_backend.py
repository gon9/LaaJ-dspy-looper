"""ローカルJSON Logger"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from laaj.observability.base import BaseObservability


class LocalLogger(BaseObservability):
    """実験結果をローカルJSONファイルに記録するLogger"""

    def __init__(self, output_dir: str | Path = "outputs"):
        """LocalLoggerを初期化

        Args:
            output_dir: 出力ディレクトリのパス
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> bool:
        """ローカルロガーの初期化（ディレクトリの作成確認）"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return True

    def log_experiment(
        self,
        experiment_name: str,
        config: dict[str, Any],
        results: dict[str, Any],
        optimized_module_path: str | None = None,
    ) -> Path:
        """実験結果をログに記録

        Args:
            experiment_name: 実験名
            config: 実験設定
            results: 実験結果
            optimized_module_path: 最適化済みモジュールの保存パス

        Returns:
            Path: ログファイルのパス
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        experiment_id = f"{experiment_name}_{timestamp}"

        log_data = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "config": config,
            "results": results,
            "optimized_module_path": optimized_module_path,
        }

        log_file = self.output_dir / f"{experiment_id}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        return log_file

    def log_evaluation(
        self,
        experiment_name: str,
        module_type: str,
        scores: dict[str, float],
        module_path: str | None = None,
    ) -> Path:
        """評価結果をログに記録

        Args:
            experiment_name: 実験名
            module_type: モジュールタイプ（baseline/optimized）
            scores: スコア辞書
            module_path: モジュールのパス

        Returns:
            Path: ログファイルのパス
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        eval_id = f"{experiment_name}_{module_type}_{timestamp}"

        log_data = {
            "eval_id": eval_id,
            "experiment_name": experiment_name,
            "module_type": module_type,
            "timestamp": timestamp,
            "scores": scores,
            "module_path": module_path,
        }

        log_file = self.output_dir / f"eval_{eval_id}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        return log_file

    def get_experiment_logs(self, experiment_name: str | None = None) -> list[dict]:
        """実験ログを取得

        Args:
            experiment_name: 実験名（指定しない場合は全て）

        Returns:
            list[dict]: 実験ログのリスト
        """
        logs = []
        for log_file in self.output_dir.glob("*.json"):
            if log_file.name.startswith("eval_"):
                continue

            with open(log_file, encoding="utf-8") as f:
                log_data = json.load(f)

            if experiment_name is None or log_data.get("experiment_name") == experiment_name:
                logs.append(log_data)

        return sorted(logs, key=lambda x: x.get("timestamp", ""), reverse=True)
