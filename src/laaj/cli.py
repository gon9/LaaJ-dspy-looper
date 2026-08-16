"""CLIエントリポイント"""

import os
from pathlib import Path

import click
import dspy

from laaj.config import load_config
from laaj.dataset import DatasetManager
from laaj.metrics import MetricRegistry
from laaj.modules import ModuleRegistry
from laaj.observability import LocalLogger
from laaj.optimizer import OptimizerEngine


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """LaaJ-dspy-looper: DSPyベース プロンプト最適化ループシステム"""
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="実験設定YAMLファイルのパス",
)
def optimize(config: str):
    """プロンプト最適化を実行"""
    click.echo(f"📋 設定ファイルを読み込み中: {config}")
    cfg = load_config(config)

    click.echo(f"🚀 実験開始: {cfg.experiment.name}")
    click.echo(f"📝 説明: {cfg.experiment.description}")

    _configure_lm(cfg.lm)

    click.echo(f"📊 データセットを読み込み中: {cfg.dataset.path}")
    dataset_manager = DatasetManager(cfg.dataset.path, cfg.dataset.input_fields)
    dataset_manager.load()
    click.echo(f"✅ データセットサイズ: {dataset_manager.size}")

    trainset, valset, testset = dataset_manager.split(
        cfg.dataset.split.train, cfg.dataset.split.val, cfg.dataset.split.test
    )
    click.echo(
        f"📦 分割: train={len(trainset)}, val={len(valset)}, test={len(testset)}"
    )

    click.echo(f"🔧 Moduleをロード中: {cfg.module.path}")
    student = ModuleRegistry.instantiate_module(cfg.module.path, cfg.module.class_name)
    click.echo(f"✅ Module: {cfg.module.class_name}")

    click.echo("📏 メトリクスを設定中...")
    metric = MetricRegistry.get_metric(
        name=cfg.metric.name,
        field=cfg.metric.field,
        custom_path=cfg.metric.custom_path,
        custom_func=cfg.metric.custom_func,
    )
    click.echo(f"✅ メトリクス: {cfg.metric.name}")

    click.echo(f"⚙️  Optimizerを作成中: {cfg.optimizer.name}")
    optimizer = OptimizerEngine.create_optimizer(cfg.optimizer, metric)
    click.echo(f"✅ Optimizer: {cfg.optimizer.name}")

    click.echo("🔄 最適化を実行中...")
    optimized_module = OptimizerEngine.optimize(
        optimizer=optimizer, student=student, trainset=trainset, valset=valset
    )
    click.echo("✅ 最適化完了")

    output_dir = Path("outputs") / cfg.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    module_path = output_dir / "optimized_module"
    optimized_module.save(str(module_path))
    click.echo(f"💾 最適化済みModuleを保存: {module_path}")

    click.echo("📊 テストセットで評価中...")
    test_score = _evaluate_module(optimized_module, testset, metric)
    click.echo(f"✅ テストスコア: {test_score:.4f}")

    logger = LocalLogger(output_dir=output_dir)
    log_file = logger.log_experiment(
        experiment_name=cfg.experiment.name,
        config=cfg.model_dump(),
        results={"test_score": test_score, "train_size": len(trainset)},
        optimized_module_path=str(module_path),
    )
    click.echo(f"📝 実験ログを保存: {log_file}")

    click.echo("🎉 最適化完了!")


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="実験設定YAMLファイルのパス",
)
@click.option(
    "--baseline", is_flag=True, help="ベースライン（最適化なし）で評価"
)
@click.option(
    "--module-path",
    "-m",
    type=click.Path(exists=True),
    help="最適化済みModuleのパス",
)
def evaluate(config: str, baseline: bool, module_path: str | None):
    """モジュールを評価"""
    click.echo(f"📋 設定ファイルを読み込み中: {config}")
    cfg = load_config(config)

    _configure_lm(cfg.lm)

    click.echo(f"📊 データセットを読み込み中: {cfg.dataset.path}")
    dataset_manager = DatasetManager(cfg.dataset.path, cfg.dataset.input_fields)
    dataset_manager.load()

    _, _, testset = dataset_manager.split(
        cfg.dataset.split.train, cfg.dataset.split.val, cfg.dataset.split.test
    )
    click.echo(f"📦 テストセットサイズ: {len(testset)}")

    click.echo("📏 メトリクスを設定中...")
    metric = MetricRegistry.get_metric(
        name=cfg.metric.name,
        field=cfg.metric.field,
        custom_path=cfg.metric.custom_path,
        custom_func=cfg.metric.custom_func,
    )

    if baseline:
        click.echo(f"🔧 ベースラインModuleをロード中: {cfg.module.path}")
        module = ModuleRegistry.instantiate_module(cfg.module.path, cfg.module.class_name)
        module_type = "baseline"
    elif module_path:
        click.echo(f"🔧 最適化済みModuleをロード中: {module_path}")
        module = ModuleRegistry.instantiate_module(cfg.module.path, cfg.module.class_name)
        module.load(module_path)
        module_type = "optimized"
    else:
        click.echo("❌ --baseline または --module-path のいずれかを指定してください")
        return

    click.echo("📊 評価中...")
    score = _evaluate_module(module, testset, metric)
    click.echo(f"✅ スコア ({module_type}): {score:.4f}")

    output_dir = Path("outputs") / cfg.experiment.name
    logger = LocalLogger(output_dir=output_dir)
    log_file = logger.log_evaluation(
        experiment_name=cfg.experiment.name,
        module_type=module_type,
        scores={"test_score": score},
        module_path=module_path,
    )
    click.echo(f"📝 評価ログを保存: {log_file}")


def _configure_lm(lm_config):
    """言語モデルを設定"""
    api_key = lm_config.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("LLM APIキーが設定されていません")

    model_str = f"{lm_config.provider}/{lm_config.model}"
    lm = dspy.LM(
        model_str,
        api_key=api_key,
        api_base=lm_config.api_base,
        temperature=lm_config.temperature,
        max_tokens=lm_config.max_tokens,
    )
    dspy.configure(lm=lm)


def _evaluate_module(module, dataset, metric):
    """モジュールを評価"""
    evaluator = dspy.Evaluate(devset=dataset, metric=metric, display_progress=True)
    score = evaluator(module)
    return score


if __name__ == "__main__":
    cli()
