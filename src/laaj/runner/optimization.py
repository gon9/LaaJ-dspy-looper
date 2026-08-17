"""最適化実行オーケストレーション Runner"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import dspy

from laaj.config import Config, load_config
from laaj.dataset import DatasetManager
from laaj.lm import LMFactory
from laaj.metrics import MetricRegistry
from laaj.modules import ModuleRegistry
from laaj.observability import ObservabilityFactory
from laaj.optimizer import OptimizerEngine


@dataclass
class OptimizationResult:
    """最適化の実行結果"""

    experiment_name: str
    test_score: float
    train_size: int
    val_size: int
    test_size: int
    optimized_module_path: str
    log_path: str | None


class OptimizationRunner:
    """プロンプト自動最適化パイプラインを実行するオーケストレーター"""

    def __init__(self, config: Config):
        self.config = config

    @classmethod
    def from_config_path(cls, config_path: str | Path) -> "OptimizationRunner":
        """設定ファイルパスからRunnerを生成"""
        cfg = load_config(config_path)
        return cls(cfg)

    def run(self, progress_callback: Callable[[str], None] | None = None) -> OptimizationResult:
        """最適化パイプラインを実行

        Args:
            progress_callback: 進行状況通知用コールバック関数

        Returns:
            OptimizationResult: 最適化結果データクラス
        """

        def log(msg: str):
            if progress_callback:
                progress_callback(msg)

        cfg = self.config
        output_dir = Path("outputs") / cfg.experiment.name
        output_dir.mkdir(parents=True, exist_ok=True)

        log(f"🚀 実験開始: {cfg.experiment.name} ({cfg.experiment.description})")

        # 1. Observabilityの初期化
        obs_backend = ObservabilityFactory.create_backend(cfg.observability, output_dir=output_dir)
        obs_backend.initialize()

        # 2. LMの設定
        log(f"🧠 言語モデルを設定中: {cfg.lm.provider}/{cfg.lm.model}")
        LMFactory.configure_default_lm(cfg.lm)

        # 3. データセットのロード & 分割
        log(f"📊 データセットを読み込み中: {cfg.dataset.path}")
        dataset_manager = DatasetManager(cfg.dataset.path, cfg.dataset.input_fields)
        dataset_manager.load()
        trainset, valset, testset = dataset_manager.split(
            cfg.dataset.split.train, cfg.dataset.split.val, cfg.dataset.split.test
        )
        log(f"📦 分割完了: train={len(trainset)}, val={len(valset)}, test={len(testset)}")

        # 4. DSPy Moduleのロード
        log(f"🔧 Moduleをロード中: {cfg.module.path} ({cfg.module.class_name})")
        student = ModuleRegistry.instantiate_module(cfg.module.path, cfg.module.class_name)

        # 5. Metricの解決
        log(f"📏 メトリクスを設定中: {cfg.metric.name}")
        metric = MetricRegistry.get_metric(
            name=cfg.metric.name,
            field=cfg.metric.field,
            custom_path=cfg.metric.custom_path,
            custom_func=cfg.metric.custom_func,
        )

        # 6. Optimizerの生成 & コンパイル
        log(f"⚙️  Optimizerを実行中: {cfg.optimizer.name}")
        optimizer = OptimizerEngine.create_optimizer(cfg.optimizer, metric)
        optimized_module = OptimizerEngine.optimize(
            optimizer=optimizer, student=student, trainset=trainset, valset=valset
        )
        log("✅ 最適化コンパイル完了")

        # 7. 最適化モジュールの永続化
        module_path = output_dir / "optimized_module.json"
        optimized_module.save(str(module_path))
        log(f"💾 最適化済みModuleを保存: {module_path}")

        # 8. テストセットでの最終評価
        log("📊 テストセットで最終評価中...")
        evaluator = dspy.Evaluate(devset=testset, metric=metric, display_progress=True)
        test_score = evaluator(optimized_module)
        log(f"✅ テストスコア: {test_score:.4f}")

        # 9. ログ記録
        results = {
            "test_score": test_score,
            "train_size": len(trainset),
            "val_size": len(valset),
            "test_size": len(testset),
        }
        log_file = obs_backend.log_experiment(
            experiment_name=cfg.experiment.name,
            config=cfg.model_dump(),
            results=results,
            optimized_module_path=str(module_path),
        )
        log(f"📝 実験ログを保存: {log_file}")

        return OptimizationResult(
            experiment_name=cfg.experiment.name,
            test_score=test_score,
            train_size=len(trainset),
            val_size=len(valset),
            test_size=len(testset),
            optimized_module_path=str(module_path),
            log_path=str(log_file) if log_file else None,
        )
