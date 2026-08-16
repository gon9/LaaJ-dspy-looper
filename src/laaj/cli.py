"""CLIエントリポイント"""

import os
from pathlib import Path

import click
import dspy

from laaj.config import load_config
from laaj.dataset import DatasetManager
from laaj.export import ModuleInspector
from laaj.metrics import MetricRegistry
from laaj.modules import ModuleRegistry
from laaj.observability import LangfuseBackend, LocalLogger
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

    _setup_observability(cfg.observability)
    _configure_lm(cfg.lm)

    click.echo(f"📊 データセットを読み込み中: {cfg.dataset.path}")
    dataset_manager = DatasetManager(cfg.dataset.path, cfg.dataset.input_fields)
    dataset_manager.load()
    click.echo(f"✅ データセットサイズ: {dataset_manager.size}")

    trainset, valset, testset = dataset_manager.split(
        cfg.dataset.split.train, cfg.dataset.split.val, cfg.dataset.split.test
    )
    click.echo(f"📦 分割: train={len(trainset)}, val={len(valset)}, test={len(testset)}")

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
    module_path = output_dir / "optimized_module.json"
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
@click.option("--baseline", is_flag=True, help="ベースライン（最適化なし）で評価")
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


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="実験設定YAMLファイルのパス",
)
@click.option(
    "--module-path",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="最適化済みModuleのパス（.json）",
)
def diff(config: str, module_path: str):
    """最適化前後のモジュール・プロンプト差分を表示"""
    click.echo(f"📋 設定ファイルを読み込み中: {config}")
    cfg = load_config(config)

    click.echo("🔍 ベースラインと最適化済みModuleを比較中...")
    base_module = ModuleRegistry.instantiate_module(cfg.module.path, cfg.module.class_name)
    opt_module = ModuleRegistry.instantiate_module(cfg.module.path, cfg.module.class_name)
    opt_module.load(module_path)

    diff_data = ModuleInspector.diff_modules(base_module, opt_module)

    click.echo("\n" + "=" * 60)
    click.echo("⚡ プロンプト最適化 差分レポート")
    click.echo("=" * 60)

    for d in diff_data["diffs"]:
        click.echo(f"\n[Predictor: {d['predictor_name']}]")
        if d["instruction_changed"]:
            click.echo("  📝 指示文 (Instruction) の変更:")
            click.echo(f"    - Before: {d['base_instruction']}")
            click.echo(f"    + After : {d['optimized_instruction']}")
        else:
            click.echo(f"  📝 指示文 (Instruction): 変更なし ({d['base_instruction']})")

        click.echo(
            f"  🎯 Few-Shot Demos: {d['base_demos_count']} 件 ➔ {d['optimized_demos_count']} 件 (差分: +{d['demos_added']} 件)"
        )
        if d["new_demos"]:
            click.echo("  💡 注入されたデモ例:")
            for i, demo in enumerate(d["new_demos"][:3], 1):
                click.echo(f"     Demo {i}: {demo}")
            if len(d["new_demos"]) > 3:
                click.echo(f"     ... 他 {len(d['new_demos']) - 3} 件")

    click.echo("\n" + "=" * 60)


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="実験設定YAMLファイルのパス",
)
@click.option(
    "--module-path",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="最適化済みModuleのパス（.json）",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "markdown"]),
    default="json",
    help="出力フォーマット",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="出力先ファイルパス（指定なしの場合は標準出力）",
)
def export(config: str, module_path: str, format: str, output: str | None):
    """最適化済みModuleのプロンプト・設定をエクスポート"""
    cfg = load_config(config)
    content = ModuleInspector.export_module(
        module_path=cfg.module.path,
        class_name=cfg.module.class_name,
        saved_path=module_path,
        output_format=format,
    )

    if output:
        Path(output).write_text(content, encoding="utf-8")
        click.echo(f"💾 エクスポート完了: {output}")
    else:
        click.echo(content)


def _setup_observability(obs_config) -> LangfuseBackend | None:
    """Observabilityバックエンドの初期化"""
    if obs_config.backend == "langfuse":
        backend = LangfuseBackend(obs_config)
        success = backend.initialize()
        if success:
            click.echo("📡 Langfuse トレーシングが有効化されました")
            return backend
        else:
            click.echo("⚠️  Langfuse の初期化をスキップしました (Local ロギングを使用)")
    return None


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
