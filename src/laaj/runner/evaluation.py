"""評価実行オーケストレーション Runner"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import dspy

from laaj.config import Config, load_config
from laaj.dataset import DatasetManager
from laaj.exceptions import ConfigError
from laaj.lm import LMFactory
from laaj.metrics import MetricRegistry
from laaj.modules import ModuleRegistry
from laaj.observability import ObservabilityFactory


@dataclass
class EvaluationResult:
    """評価の実行結果"""

    experiment_name: str
    module_type: str
    test_score: float
    test_size: int
    module_path: str | None
    log_path: str | None


class EvaluationRunner:
    """モジュールの評価を実行するオーケストレーター"""

    def __init__(self, config: Config):
        self.config = config

    @classmethod
    def from_config_path(cls, config_path: str | Path) -> "EvaluationRunner":
        """設定ファイルパスからRunnerを生成"""
        cfg = load_config(config_path)
        return cls(cfg)

    def run(
        self,
        baseline: bool = False,
        module_path: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> EvaluationResult:
        """評価を実行

        Args:
            baseline: ベースライン（未最適化）モジュールを評価するか
            module_path: 最適化済みモジュールの保存パス
            progress_callback: 進行状況通知用コールバック

        Returns:
            EvaluationResult: 評価結果データクラス

        Raises:
            ConfigError: baseline と module_path の双方が指定されていない場合
        """

        def log(msg: str):
            if progress_callback:
                progress_callback(msg)

        if not baseline and not module_path:
            raise ConfigError("--baseline または --module-path のいずれかを指定してください")

        cfg = self.config
        output_dir = Path("outputs") / cfg.experiment.name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Observability
        obs_backend = ObservabilityFactory.create_backend(cfg.observability, output_dir=output_dir)
        obs_backend.initialize()

        # 2. LM
        log(f"🧠 言語モデルを設定中: {cfg.lm.provider}/{cfg.lm.model}")
        LMFactory.configure_default_lm(cfg.lm)

        # 3. データセット
        log(f"📊 データセットを読み込み中: {cfg.dataset.path}")
        dataset_manager = DatasetManager(cfg.dataset.path, cfg.dataset.input_fields)
        dataset_manager.load()
        _, _, testset = dataset_manager.split(
            cfg.dataset.split.train, cfg.dataset.split.val, cfg.dataset.split.test
        )
        log(f"📦 テストセットサイズ: {len(testset)}")

        # 4. Metric
        metric = MetricRegistry.get_metric(
            name=cfg.metric.name,
            field=cfg.metric.field,
            custom_path=cfg.metric.custom_path,
            custom_func=cfg.metric.custom_func,
        )

        # 5. Module
        if baseline:
            log(f"🔧 ベースラインModuleをロード中: {cfg.module.path} ({cfg.module.class_name})")
            module = ModuleRegistry.instantiate_module(cfg.module.path, cfg.module.class_name)
            module_type = "baseline"
        else:
            log(f"🔧 最適化済みModuleをロード中: {module_path}")
            module = ModuleRegistry.instantiate_module(cfg.module.path, cfg.module.class_name)
            module.load(module_path)
            module_type = "optimized"

        # 6. 評価実行
        log("📊 評価中...")
        evaluator = dspy.Evaluate(devset=testset, metric=metric, display_progress=True)
        score = evaluator(module)
        log(f"✅ スコア ({module_type}): {score:.4f}")

        # 7. ログ記録
        log_file = obs_backend.log_evaluation(
            experiment_name=cfg.experiment.name,
            module_type=module_type,
            scores={"test_score": score},
            module_path=module_path,
        )
        log(f"📝 評価ログを保存: {log_file}")

        return EvaluationResult(
            experiment_name=cfg.experiment.name,
            module_type=module_type,
            test_score=score,
            test_size=len(testset),
            module_path=module_path,
            log_path=str(log_file) if log_file else None,
        )
