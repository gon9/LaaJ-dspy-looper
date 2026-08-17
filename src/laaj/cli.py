"""CLIエントリポイント"""

import sys

import click

from laaj.exceptions import LaaJError
from laaj.runner import DiffRunner, EvaluationRunner, ExportRunner, OptimizationRunner


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
    try:
        runner = OptimizationRunner.from_config_path(config)
        result = runner.run(progress_callback=click.echo)
        click.echo(f"\n🎉 最適化が正常に完了しました! (最終テストスコア: {result.test_score:.4f})")
    except LaaJError as e:
        click.echo(f"❌ エラーが発生しました: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ 予期しないエラーが発生しました: {e}", err=True)
        sys.exit(1)


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
    help="最適化済みModuleのパス（.json）",
)
def evaluate(config: str, baseline: bool, module_path: str | None):
    """モジュールを評価"""
    if not baseline and not module_path:
        click.echo("❌ --baseline または --module-path のいずれかを指定してください", err=True)
        return

    try:
        runner = EvaluationRunner.from_config_path(config)
        result = runner.run(
            baseline=baseline,
            module_path=module_path,
            progress_callback=click.echo,
        )
        click.echo(f"✅ 評価完了: スコア ({result.module_type}) = {result.test_score:.4f}")
    except LaaJError as e:
        click.echo(f"❌ エラーが発生しました: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ 予期しないエラーが発生しました: {e}", err=True)
        sys.exit(1)


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
    try:
        runner = DiffRunner.from_config_path(config)
        diff_data = runner.run(module_path=module_path)

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

    except Exception as e:
        click.echo(f"❌ 差分抽出中にエラーが発生しました: {e}", err=True)
        sys.exit(1)


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
    try:
        runner = ExportRunner.from_config_path(config)
        content = runner.run(
            module_path=module_path,
            output_format=format,
            output_file=output,
        )
        if output:
            click.echo(f"💾 エクスポート完了: {output}")
        else:
            click.echo(content)
    except Exception as e:
        click.echo(f"❌ エクスポート中にエラーが発生しました: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
