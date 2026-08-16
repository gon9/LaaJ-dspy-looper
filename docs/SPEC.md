# LaaJ-dspy-looper: DSPyベース プロンプト最適化ループシステム 仕様書

## 1. 背景と課題

### 1.1 現状の課題

- **プロンプトチューニングが手動で非効率**: 手作業でのプロンプト調整は試行錯誤の繰り返しであり、再現性が低い。微小な変更（トークンの追加・削除・言い換え）でもLLMの性能に大きく影響する。
- **LLM-as-a-Judge単体では不十分**: 社内にLLM-as-a-Judgeの仕組みはあるが、最終的にアウトプットを目視確認する必要があり、最適化ループが閉じていない。
- **評価→改善のサイクルが断絶**: 評価結果をプロンプト改善に自動でフィードバックする仕組みがない。

### 1.2 目指す姿

評価メトリクスに基づいてプロンプトを**アルゴリズム的に最適化**し、Human-in-the-loopは「最終確認」に集中できる状態を作る。

---

## 2. 事例調査レポート

### 2.1 LayerX（バクラク事業部）

**出典**: [AI Agent時代における「使えば使うほど賢くなるAI機能」の開発](https://tech.layerx.co.jp/entry/2025/10/23/222742)

**概要**:
- 経費精算の承認/否認を判定するAgentをDSPyで構築し、Prompt最適化の効果を検証。
- 擬似データセット1000件を作成し、3つのOptimizer（BootstrapFewShotWithRandomSearch, KNNFewShot, MIPROv2）を比較。
- GPT-4.1-miniで全実験を約1時間で完了。

**主要な知見**:
- MIPROv2はInstruction最適化も同時に行い、最も劇的にプロンプトが変化した。元の簡素な指示文が、具体的な審査基準を含む詳細なプロンプトに自動変換された。
- Few-shot系はSystem messageの後に例が追加される形式。
- 大量データでの学習はコスト的に厳しいため、**局所的なパーソナライゼーション**（顧客ごとの業務ルール適応等）が適切な利用箇所。
- Context Engineering（プロンプトに渡す情報の設計）とDSPyの組み合わせが重要。

### 2.2 note株式会社

**出典**: [DSPyで始めるプロンプト最適化](https://note.com/k_urushi/n/n5eb7fea0401c)

**概要**:
- Amazon Bedrockを活用した推薦チームでDSPyを検証。
- PyTorchライクな小さいModuleを組み合わせる設計を高評価。

**主要な知見**:
- DSPyのOptimizerは3カテゴリに分類される:
  1. **Few-Shot Learning**: プロンプトに含める例を最適化（BootstrapFewShot, KNNFewShot）
  2. **Instruction Optimization**: 指示文を最適化（MIPROv2, COPRO）
  3. **Finetuning**: モデルの重み更新
- データ量によるOptimizer選択ガイドラインが有用:
  - 10件程度 → BootstrapFewShot
  - 50件以上 → BootstrapFewShotWithRandomSearch
  - 200件以上＋長時間許容 → MIPROv2
- metricは `def metric(example, pred, trace=None) -> float` のインタフェース。

### 2.3 Jina AI

**出典**: [DSPy：通常のプロンプトエンジニアリングとは一線を画す](https://jina.ai/ja/news/dspy-not-your-average-prompt-engineering/)

**主要な知見**:
- DSPyの革新は「プロンプトエンジニアリングのループを閉じる」こと。評価メトリクスを得た後、プロンプトをどう改善するかを自動化。
- **ロジックとテキスト表現の分離**: `dspy.Module`でロジックをプログラミングし、テキスト表現（プロンプト）はDSPyが最適化。
- metric関数は`trace=None`の値によって**評価関数と損失関数の二重の役割**を持つ。
- metric関数の設計はModule設計と同程度に重要で、最適化結果に直結する。

### 2.4 arXiv論文: Multi-Use Case Study

**出典**: [Is It Time To Treat Prompts As Code?](https://arxiv.org/html/2507.03620v1)

**検証ユースケース**:
1. Jailbreak検出
2. Pandasコードのハルシネーション検出
3. Pandasコード生成Agent
4. ルーティングAgent
5. プロンプト評価器

**主要な知見**:
- DSPyの構造だけで（最適化なしでも）手動プロンプトより性能が向上するケースがあった。
- 最適化済みプロンプトをDSPy外に抽出して使うと性能が劣化する可能性あり（DSPyの内部推論挙動に依存するため）。
- **DSPyはフルプログラミングモデルとして使うべき**。

### 2.5 Langfuse連携

**出典**: [Langfuse DSPy Integration](https://langfuse.com/integrations/frameworks/dspy)

**連携方法**:
- `openinference-instrumentation-dspy` を使用してOpenTelemetryベースのトレーシングを実現。
- `DSPyInstrumentor().instrument()` の1行でLM呼び出しを自動キャプチャ。
- Langfuseの`@observe()`デコレータや`propagate_attributes`で追加メタデータ（user_id, session_id, tags等）も付与可能。

---

## 3. DSPy利用用途の整理

| 用途 | 説明 | 適切なOptimizer |
|------|------|----------------|
| **分類タスクの精度向上** | テキスト分類、感情分析等 | BootstrapFewShot, MIPROv2 |
| **判定タスクの自動化** | 承認/否認、合否判定等 | BootstrapFewShotWithRandomSearch, MIPROv2 |
| **RAGパイプラインの最適化** | 検索+生成の品質向上 | MIPROv2 |
| **Guardrails最適化** | Jailbreak・ハルシネーション検出 | BootstrapFewShot |
| **Agent/ワークフロー最適化** | マルチステップのAgent処理 | MIPROv2 |
| **パーソナライゼーション** | 顧客/ユースケース別の適応 | KNNFewShot |

### 本プロジェクトでのフォーカス

本ツール「LaaJ-dspy-looper」は、以下のユースケースをターゲットとする:

1. **汎用プロンプト最適化ループ**: 任意のLLMタスク（分類、生成、判定等）に対してDSPyのOptimizerを適用し、評価→最適化のサイクルを回す。
2. **LLM-as-a-Judge連携**: 既存の社内LLM-as-a-Judge評価基盤とmetric関数を接続し、評価結果を最適化にフィードバック。
3. **実験管理・ログ**: Langfuse（またはMLflow）を用いたトレーシングで、最適化過程と結果を記録・比較可能にする。

---

## 4. システム設計

### 4.1 アーキテクチャ概要

```
┌─────────────────────────────────────────────────────────┐
│                    LaaJ-dspy-looper                      │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │ Dataset   │──▶│ DSPy Module  │──▶│ Optimizer      │  │
│  │ Manager   │   │ (Signature/  │   │ (MIPRO/BFRS/   │  │
│  │           │   │  Module)     │   │  KNN)          │  │
│  └──────────┘   └──────────────┘   └───────┬────────┘  │
│                                             │           │
│  ┌──────────┐   ┌──────────────┐           │           │
│  │ Metric   │──▶│ Evaluator    │◀──────────┘           │
│  │ Registry │   │              │                        │
│  └──────────┘   └──────┬───────┘                        │
│                         │                               │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │           Observability Layer                     │   │
│  │  (Langfuse / MLflow / Local JSON Logger)          │   │
│  └───────────────────────────────────────────────────┘   │
│                                                         │
│  ┌───────────────────────────────────────────────────┐   │
│  │           CLI / Config Interface                   │   │
│  │  (YAML config → experiment定義 → 実行 → レポート)   │   │
│  └───────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 4.2 コンポーネント詳細

#### 4.2.1 Dataset Manager

- **役割**: 学習/検証/テストデータの管理。
- **入力フォーマット**: JSONL。各行は `dspy.Example` に変換可能なキー・バリュー形式。
- **機能**:
  - データの読み込み・分割（train/val/test）
  - `dspy.Example` への変換
  - `with_inputs()` による入力フィールド指定

```python
# データフォーマット例 (JSONL)
{"input_text": "...", "expected_output": "approved", "metadata": {...}}
```

#### 4.2.2 DSPy Module Registry

- **役割**: Signature/Moduleの定義と管理。
- **設計方針**: ユーザが自身のタスクに合わせてSignatureとModuleをPythonコードで定義。プロジェクトの `modules/` ディレクトリに配置。
- **組み込みModule**: ChainOfThought, Predict, ReAct等のDSPy標準Moduleをラップ。

```python
# ユーザ定義のSignature例
class MyTaskSignature(dspy.Signature):
    """タスクの説明をここに記載"""
    input_text: str = dspy.InputField(desc="入力テキスト")
    output_label: str = dspy.OutputField(desc="出力ラベル")
```

#### 4.2.3 Metric Registry

- **役割**: 評価メトリクスの定義・管理。
- **組み込みメトリクス**:
  - `exact_match`: 完全一致
  - `llm_as_judge`: LLMによる評価（社内ツール連携可能）
  - `semantic_similarity`: 意味的類似度
- **カスタムメトリクス**: ユーザがPython関数として定義。`def metric(example, pred, trace=None) -> float` のインタフェースに準拠。

```python
# LLM-as-a-Judge metric例
def llm_judge_metric(example, pred, trace=None):
    """LLMを使った評価メトリクス"""
    judge = dspy.Predict("gold_answer, prediction -> score: float")
    result = judge(gold_answer=example.expected_output, prediction=pred.output_label)
    score = float(result.score)
    if trace is not None:
        return score  # 学習時: 連続値スコア
    return score > 0.7  # 評価時: 閾値判定
```

#### 4.2.4 Optimizer Engine

- **役割**: DSPyのOptimizerをラップし、設定ファイルから実行可能にする。
- **サポートするOptimizer**:

| Optimizer | 用途 | 主要パラメータ |
|-----------|------|---------------|
| `BootstrapFewShot` | 少量データのFew-shot最適化 | `max_bootstrapped_demos`, `max_labeled_demos` |
| `BootstrapFewShotWithRandomSearch` | 中量データのFew-shot最適化 | 上記 + `num_candidate_programs` |
| `KNNFewShot` | 類似例ベースのFew-shot | `k`, `vectorizer` |
| `MIPROv2` | Instruction + Few-shot最適化 | `num_candidates`, `num_trials`, `init_temperature` |
| `COPRO` | シンプルなInstruction最適化 | - |

#### 4.2.5 Observability Layer

- **役割**: 実験のトレーシング・ログ記録。
- **バックエンド（選択可能）**:
  1. **Langfuse**: OpenTelemetry + `openinference-instrumentation-dspy` 経由。最適化過程のLM呼び出しを自動キャプチャ。
  2. **Local JSON Logger**: Langfuseを使わない場合のフォールバック。実験結果をJSONファイルに保存。
- **記録する情報**:
  - 実験ID、タイムスタンプ
  - 使用したOptimizer・パラメータ
  - 最適化前後のプロンプト（diff）
  - train/val/testスコア
  - LM呼び出し回数・コスト概算
  - 最適化済みModuleのスナップショット（`module.save()`）

#### 4.2.6 CLI / Config Interface

- **実験定義**: YAML設定ファイルで実験を宣言的に記述。

```yaml
# experiments/my_experiment.yaml
experiment:
  name: "expense_approval_optimization"
  description: "経費精算承認タスクのプロンプト最適化"

dataset:
  path: "data/expense_approval.jsonl"
  input_fields: ["expense_form"]
  split:
    train: 0.7
    val: 0.15
    test: 0.15

module:
  path: "modules/expense_approval.py"
  class_name: "ExpenseApprovalModule"

lm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 1.0
  max_tokens: 16000

optimizer:
  name: "MIPROv2"
  params:
    num_candidates: 5
    num_trials: 10
    max_bootstrapped_demos: 4
    max_labeled_demos: 10
    init_temperature: 1.0

metric:
  name: "exact_match"
  field: "approval_status"
  # または
  # custom_path: "metrics/my_metric.py"
  # custom_func: "my_custom_metric"

observability:
  backend: "langfuse"  # or "local"
  # langfuse_public_key: env:LANGFUSE_PUBLIC_KEY
  # langfuse_secret_key: env:LANGFUSE_SECRET_KEY
```

- **CLIコマンド**:

```bash
# 最適化の実行
laaj optimize --config experiments/my_experiment.yaml

# ベースラインの評価（最適化なし）
laaj evaluate --config experiments/my_experiment.yaml --baseline

# 最適化済みModuleでの評価
laaj evaluate --config experiments/my_experiment.yaml --module-path outputs/optimized_module/

# 最適化前後のプロンプト比較
laaj diff --config experiments/my_experiment.yaml --module-path outputs/optimized_module/

# 最適化済みModuleのエクスポート
laaj export --module-path outputs/optimized_module/ --format json
```

---

## 5. プロジェクト構成

```
LaaJ-dspy-looper/
├── pyproject.toml                  # uv用プロジェクト定義
├── Dockerfile                      # マルチステージビルド
├── README.md
├── docs/
│   └── SPEC.md                     # 本仕様書
├── src/
│   └── laaj/
│       ├── __init__.py
│       ├── cli.py                  # CLIエントリポイント
│       ├── config.py               # YAML設定の読み込み・バリデーション
│       ├── dataset/
│       │   ├── __init__.py
│       │   └── manager.py          # Dataset Manager
│       ├── modules/
│       │   ├── __init__.py
│       │   ├── registry.py         # Module Registry
│       │   └── builtin.py          # 組み込みModule
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── registry.py         # Metric Registry
│       │   ├── exact_match.py
│       │   ├── llm_judge.py
│       │   └── semantic_similarity.py
│       ├── optimizer/
│       │   ├── __init__.py
│       │   └── engine.py           # Optimizer Engine
│       ├── observability/
│       │   ├── __init__.py
│       │   ├── langfuse_backend.py
│       │   └── local_backend.py
│       └── export/
│           ├── __init__.py
│           └── exporter.py         # 最適化結果のエクスポート
├── experiments/                    # 実験設定YAML
│   └── example.yaml
├── data/                           # データセット
│   └── .gitkeep
├── outputs/                        # 最適化結果
│   └── .gitkeep
└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_dataset_manager.py
    ├── test_metrics.py
    ├── test_optimizer_engine.py
    └── test_observability.py
```

---

## 6. 技術スタック

| カテゴリ | 技術 | バージョン |
|---------|------|-----------|
| 言語 | Python | 3.12 |
| パッケージ管理 | uv | latest |
| LLM最適化 | dspy | >= 2.6 |
| LLM呼び出し | litellm（dspy内部） | - |
| CLI | click or typer | latest |
| 設定管理 | pydantic + PyYAML | latest |
| Observability | langfuse + openinference-instrumentation-dspy | latest |
| Linter/Formatter | ruff | latest |
| テスト | pytest | latest |
| コンテナ | Docker（マルチステージビルド） | - |

---

## 7. 実装フェーズ

### Phase 1: 基盤構築（MVP）

**ゴール**: 単一タスクに対してDSPy最適化を実行し、結果をローカルに保存できる。

- [ ] プロジェクトセットアップ（uv, pyproject.toml, ruff設定）
- [ ] Config（YAML → Pydanticモデル）
- [ ] Dataset Manager（JSONL読み込み → dspy.Example変換）
- [ ] Optimizer Engine（MIPROv2 + BootstrapFewShotWithRandomSearch）
- [ ] 組み込みMetric（exact_match）
- [ ] Local JSON Logger
- [ ] CLI基本コマンド（optimize, evaluate）
- [ ] 基本テスト

### Phase 2: 評価・メトリクス拡充

- [ ] LLM-as-a-Judge metric
- [ ] Semantic Similarity metric
- [ ] カスタムMetricのプラグイン機構
- [ ] diff コマンド（最適化前後のプロンプト比較）
- [ ] テスト拡充

### Phase 3: Observability統合

- [ ] Langfuseバックエンド
- [ ] 実験メタデータの自動記録
- [ ] Langfuse上での実験比較ダッシュボード活用
- [ ] テスト拡充

### Phase 4: プロダクション対応

- [ ] Dockerfileマルチステージビルド
- [ ] 環境変数管理（.env / secrets）
- [ ] エラーハンドリング・リトライ
- [ ] export コマンド
- [ ] ドキュメント整備

---

## 8. 設計上の判断ポイント

### 8.1 Langfuseは必要か？

**結論: Phase 1では不要、Phase 3で導入。**

**理由**:
- DSPy自体が`dspy.inspect_history()`で実行プロンプトを確認できる。
- `module.save()` / `module.load()` で最適化結果の永続化が可能。
- Local JSON Loggerでも実験結果の記録・比較は十分可能。
- Langfuseは「チームでの実験共有」「長期的なトレンド分析」「既存のLLM監視基盤との統合」で真価を発揮するため、基盤が安定してから導入する。

### 8.2 DSPyの外でプロンプトを使うべきか？

**結論: DSPy内で完結させることを推奨。**

**理由**:
- arXiv論文の知見として、最適化済みプロンプトをDSPy外に抽出すると性能が劣化する可能性がある。
- DSPyの内部推論挙動（フォーマット、パース等）に最適化結果が依存するため。
- ただし、Instructionのみの抽出（MIPROv2で最適化した指示文）は参考情報として有用。`export`コマンドでJSON出力する機能は提供する。

### 8.3 Optimizerの選定基準

```
データ量少（~10件）  → BootstrapFewShot
データ量中（~50件）  → BootstrapFewShotWithRandomSearch
データ量多（200件~） → MIPROv2
類似例活用          → KNNFewShot
コスト最小化        → COPRO（Instruction最適化のみ）
```

### 8.4 metric関数のtrace引数の扱い

DSPyのmetric関数は `trace` 引数で評価/学習を切り替える設計:
- `trace is not None`（学習時）: 連続値スコアを返す（損失関数的役割）
- `trace is None`（評価時）: bool/閾値判定を返す

この設計はDSPyの最適化品質に直結するため、ドキュメントとサンプルで明確にガイドする。

---

## 9. 制約・前提条件

- LLM APIキーはユーザが自身で用意する（OpenAI, Azure OpenAI, Anthropic等、litellm対応プロバイダ）。
- データセットはユーザが自身で用意する（JSONL形式）。
- DSPyの最適化にはLLM API呼び出しコストが発生する（MIPROv2 + GPT-4o-miniで数百〜数千回の呼び出し）。
- 最適化はバッチ処理であり、リアルタイム最適化は対象外。

---

## 10. 成功指標

1. **最適化の効果が定量的に確認できる**: ベースライン vs 最適化後のスコア比較がCLI上で即座に確認可能。
2. **実験の再現性**: YAML設定 + データセットがあれば同一実験を再現可能。
3. **低い導入コスト**: Signature/Module/Metricの定義 + YAML設定のみで最適化を開始できる。
4. **段階的に機能拡張可能**: Phase 1のMVPから始めて、Observabilityやカスタムメトリクスを段階的に追加可能。
