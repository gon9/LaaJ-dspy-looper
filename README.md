# LaaJ-dspy-looper

DSPyベース プロンプト最適化ループシステム

## 概要

LaaJ-dspy-looperは、DSPyフレームワークを活用してLLMプロンプトをアルゴリズム的に最適化するツールです。
手動でのプロンプトチューニングから脱却し、評価メトリクスに基づいた自動最適化を実現します。

## 主な機能

- **YAML設定ファイル駆動**: 実験を宣言的に定義し、CLIから実行
- **複数のOptimizer対応**: MIPROv2, BootstrapFewShotWithRandomSearch, KNNFewShot等
- **カスタムメトリクス**: LLM-as-a-Judge含む柔軟な評価関数
- **実験管理**: ローカルJSON Logger（Phase 3でLangfuse対応予定）

## セットアップ

### 前提条件

- Python 3.12以上
- uv（パッケージマネージャ）

### インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd LaaJ-dspy-looper

# 依存関係のインストール
uv sync

# 開発用依存関係も含める場合
uv sync --all-extras
```

## クイックスタート

### 1. データセットの準備

JSONL形式でデータセットを作成します:

```jsonl
{"input_text": "質問文1", "expected_output": "回答1"}
{"input_text": "質問文2", "expected_output": "回答2"}
```

### 2. 実験設定の作成

`experiments/my_experiment.yaml` を作成:

```yaml
experiment:
  name: "my_first_optimization"
  description: "最初のプロンプト最適化実験"

dataset:
  path: "data/my_dataset.jsonl"
  input_fields: ["input_text"]
  split:
    train: 0.7
    val: 0.15
    test: 0.15

module:
  path: "modules/my_module.py"
  class_name: "MyModule"

lm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 1.0

optimizer:
  name: "MIPROv2"
  params:
    num_candidates: 5
    num_trials: 10

metric:
  name: "exact_match"
  field: "output"

observability:
  backend: "local"
```

### 3. 最適化の実行

```bash
# 最適化実行
laaj optimize --config experiments/my_experiment.yaml

# ベースライン評価
laaj evaluate --config experiments/my_experiment.yaml --baseline

# 最適化済みモジュールでの評価
laaj evaluate --config experiments/my_experiment.yaml --module-path outputs/optimized_module/
```

## プロジェクト構成

```
LaaJ-dspy-looper/
├── src/laaj/          # メインパッケージ
├── experiments/       # 実験設定YAML
├── data/             # データセット
├── outputs/          # 最適化結果
└── tests/            # テスト
```

## ドキュメント

- [docs/INDEX.md](docs/INDEX.md): ドキュメント目次
- [docs/SPEC.md](docs/SPEC.md): 詳細仕様書・事例調査（LayerX, note, Jina AI 等）
- [docs/DSPY_OVERVIEW.md](docs/DSPY_OVERVIEW.md): DSPy 概要・基本概念ガイド
- [docs/ENABLEMENT_PLAN.md](docs/ENABLEMENT_PLAN.md): イネーブルメント（学習・チーム普及）計画


## ライセンス

MIT License
