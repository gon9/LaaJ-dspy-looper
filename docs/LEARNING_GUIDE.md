# DSPy 体系的学習マスターガイド (Learning Master Guide - 完全版)

> **対象**: LLM アプリケーション開発者、プロンプト最適化を自動化したいエンジニア、AI Agent アーキテクト  
> **ゴール**: 評価駆動開発（Eval-Driven Development）と自動プロンプト最適化ループを自力で設計・構築・運用できるようになる

---

## 📑 目次

- [Chapter 1: DSPy メンタルモデル & 思考のパラダイムシフト](#chapter-1-dspy-メンタルモデル--思考のパラダイムシフト)
- [Chapter 2: 入出力設計（Signature）& モジュール構築（Module）](#chapter-2-入出力設計signature--モジュール構築module)
- [Chapter 3: 評価関数（Metric）エンジニアリング & trace 引数](#chapter-3-評価関数metricエンジニアリング--trace-引数)
- [Chapter 4: 自動最適化（Optimizer）の実践 (BootstrapFewShot & MIPROv2)](#chapter-4-自動最適化optimizerの実践-bootstrapfewshot--miprov2)
- [Chapter 5: LLM-as-a-Judge, 差分比較 & Observability (可視化)](#chapter-5-llm-as-a-judge-差分比較--observability-可視化)
- [Chapter 6: 高度な技術深掘り（Assertions, Pydantic, 過学習対策）](#chapter-6-高度な技術深掘りassertions-pydantic-過学習対策)
- [Chapter 7: コミュニティ最新動向 & 次世代技術 (GEPA, ReActV2, BetterTogether)](#chapter-7-コミュニティ最新動向--次世代技術-gepa-reactv2-bettertogether)
- [Practice: ハンズオン実践課題集 & 解答例](#practice-ハンズオン実践課題集--解答例)

---

# Chapter 1: DSPy メンタルモデル & 思考のパラダイムシフト

## 1. 従来のプロンプトエンジニアリングの限界

従来のプロンプト開発では、人間が手作業で以下を行っていました：
- 「あなたは親切なアシスタントです...」のような指示文（Instruction）の試行錯誤
- 「例えば以下のように出力してください...」という Few-Shot 例の手動選定
- モデルの機嫌を取るための文末調整やキーワードの追加

### 発生する課題（Prompt Decay: プロンプト劣化）
手作業で調整したプロンプトは、基盤モデル（例: GPT-4o ➔ Claude 3.5 ➔ Gemini 3.7 ➔ ローカルLLM）を変更した瞬間、またはタスクの微小な仕様変更があった瞬間に **精度が急激に低下（劣化）** します。

```mermaid
flowchart LR
    subgraph Traditional["従来の開発"]
        A1[タスク変更 / モデル変更] --> B1[プロンプトを手動で書き直し]
        B1 --> C1[目視確認]
        C1 -->|精度低下| B1
    end

    subgraph DSPy["DSPy による開発"]
        A2[タスク変更 / モデル変更] --> B2[型と評価関数をそのまま保持]
        B2 --> C2[Optimizer を再実行 (自動探索)]
        C2 --> D2[新モデルに最適なプロンプトが自動生成]
    end
```

---

## 2. PyTorch と DSPy のメンタルモデル対比

DSPy (Declarative Self-improving Python) は、**PyTorch の設計思想を LLM プログラミングに適用したもの** です。

| 開発要素 | PyTorch (機械学習) | DSPy (LLMプログラミング) | 担当 |
| :--- | :--- | :--- | :--- |
| **タスク契約** | Tensor の shape (`torch.Size([B, D])`) | **`dspy.Signature`** (`input -> output`) | **🧑‍💻 開発者** |
| **処理構造** | `nn.Module` (Linear, Conv 等の結合) | **`dspy.Module`** (`Predict`, `ChainOfThought` 等) | **🧑‍💻 開発者** |
| **評価基準** | `Loss Function` (MSE, CrossEntropy 等) | **`Metric`** (EM, LLM-as-a-Judge 等) | **🧑‍💻 開発者** |
| **最適化対象** | モデルの重み・パラメータ (Weights) | **指示文 (Instructions) と Few-Shot 例 (Demos)** | **🤖 Optimizer** |
| **探索手法** | 勾配降下法 (SGD, Adam 等) | **Bootstrap, ベイズ最適化, 遺伝的アルゴリズム** | **🤖 Optimizer** |

---

## 3. 開発者とアルゴリズムの明確な役割分担

> [!IMPORTANT]
> **黄金律**: 開発者は「**何を入力して何を出すか（型）**」と「**どういう処理の流れか（構造）**」だけをコードで定義する。
> 「**どんな文言で指示するか**」「**どんな例を載せるか**」は **アルゴリズム（Optimizer）に完全委譲する**。

```mermaid
flowchart TD
    subgraph Dev["🧑‍💻 開発者が書くもの"]
        D1["1. dspy.Signature (入出力の型契約)"]
        D2["2. dspy.Module (処理の組み立て: CoT, ReAct等)"]
        D3["3. Metric (出力の良し悪しを判定する評価関数)"]
        D4["4. Dataset (評価用データセット)"]
    end

    subgraph Algo["🤖 Optimizer が自動生成・探索するもの"]
        A1["1. 指示文 (Instruction: モデルが最も理解しやすい言葉遣い)"]
        A2["2. Few-Shot 例 (Demos: 成功トレースから選別した最良の例)"]
        A3["3. プロンプト全体の最適構造"]
    end

    Dev -->|入力| Algo
```

---

# Chapter 2: 入出力設計（Signature）& モジュール構築（Module）

## 1. Signature の定義: 2つのスタイル

```python
import dspy

# クラスベース定義（推奨）
class ExpenseApprovalSignature(dspy.Signature):
    """提出された経費申請内容を審査し、承認(approved)または否認(rejected)を判定する。"""

    expense_details: str = dspy.InputField(desc="経費申請の詳細（品目、金額、理由等）")
    approval_status: str = dspy.OutputField(desc="approved または rejected")
```

---

## 2. 推論モジュール（Modules）の選定と使い分け

| モジュール | 動作 | 適用ユースケース | コスト / 速度 |
| :--- | :--- | :--- | :--- |
| **`dspy.Predict`** | 入力から直接出力を生成 | 単純分類、エンティティ抽出、短い定型変換 | 最速・最安 (トークン消費小) |
| **`dspy.ChainOfThought`** | `reasoning` フィールドを前挿入して思考 | 論理的推論、複数条件の判定、要約、審査 | 高精度 (+5〜15% 向上), 中速 |
| **`dspy.ReAct`** | Thought ➔ Action (Tool) ➔ Observation ループ | 外部検索、コード実行、Web検索を伴うAgent | 高機能, 複数回LLM呼出 |

```mermaid
flowchart TD
    subgraph Predict["dspy.Predict"]
        P_IN[Input: 経費情報] --> P_OUT[Output: 判定結果]
    end

    subgraph CoT["dspy.ChainOfThought"]
        C_IN[Input: 経費情報] --> C_THINK["Reasoning: なぜその判定になるか思考"]
        C_THINK --> C_OUT[Output: 判定結果]
    end
```

---

# Chapter 3: 評価関数（Metric）エンジニアリング & trace 引数

## 1. データセットの準備と `dspy.Example`

```python
import dspy

example = dspy.Example(
    expense_details="クライアントとの懇親会飲食代 15,000円 (参加者3名、事前伺い済)",
    expected_status="approved",
    department="営業部"
).with_inputs("expense_details")
```

---

## 2. Metric 関数のインターフェースと `trace` 引数の二重性

| 実行フェーズ | `trace` の値 | 役割 | 推奨される戻り値 |
| :--- | :--- | :--- | :--- |
| **学習時 (Optimization中)** | `trace is not None` | **損失関数 (Loss Function)** | **連続値スコア（部分点あり）**<br>例: 0.0, 0.5, 0.8, 1.0 |
| **評価時 (Test/Validation)** | `trace is None` | **評価指標 (Evaluation Metric)** | **厳格な判定スコア**<br>例: 1.0 (合格) / 0.0 (不合格) |

```mermaid
flowchart TD
    M[Metric 関数呼び出し] --> C{trace 引数の判定}
    C -->|trace is not None<br>Optimizer 探索中| L[損失関数モード<br>・部分点を付与<br>・探索勾配を作る]
    C -->|trace is None<br>テスト・検証中| E[評価指標モード<br>・厳格な判定 (1.0 or 0.0)<br>・正確な集計]
```

---

# Chapter 4: 自動最適化（Optimizer）の実践 (BootstrapFewShot & MIPROv2)

## 1. Optimizer の選定マトリクス

| データ量目安 | 推奨 Optimizer | 最適化対象 | 特徴・使いどころ |
| :--- | :--- | :--- | :--- |
| **〜10件** | `BootstrapFewShot` | Few-Shot デモ例 | 最速・低コスト。成功した実行トレースをプロンプトに埋め込む。 |
| **10〜50件** | `BootstrapFewShotWithRandomSearch` | Few-Shot デモ例 | 複数パターンのデモ候補をランダム探索し、最も高スコアな組み合わせを選択。 |
| **50件〜200件以上** | **`MIPROv2`** (業界標準) | **指示文 + Few-Shot** | **Meta-LLM による指示文自動生成 ＋ Optuna ベイズ最適化**。最大の性能向上。 |
| **類似検索** | `KNNFewShot` | Few-Shot デモ例 | 入力テキストに意味的に最も近い事例を実行時に動的取得（埋め込み利用）。 |
| **指示文のみ** | `COPRO` | 指示文 (Instruction) | Few-Shot を増やさずにプロンプトを短く保ちたい場合。 |

---

## 2. `MIPROv2` の内部メカニズム

```mermaid
flowchart LR
    A[タスク Signature & データ] --> B[Meta-LLM<br>多様な指示文候補を複数生成]
    A --> C[Bootstrapper<br>Few-Shot デモ候補プール生成]
    B --> D[Optuna TPE<br>ベイズ最適化探索空間]
    C --> D
    D --> E[ミニバッチ評価<br>有望な組み合わせを絞り込み]
    E --> F[大バッチ検証<br>最高スコアのモジュールを出力]
```

---

# Chapter 5: LLM-as-a-Judge, 差分比較 & Observability (可視化)

## 1. プロンプト差分の確認 (`laaj diff`)

```bash
uv run laaj diff --config experiments/example.yaml --module-path outputs/example_classification/optimized_module.json
```

```text
============================================================
⚡ プロンプト最適化 差分レポート
============================================================

[Predictor: classifier]
  📝 指示文 (Instruction) の変更:
    - Before: テキストを分類するタスク
    + After : 顧客の文脈とトーンを考慮し、不満やトラブルに関する発言は negative、感謝や満足は positive と分類する。
  🎯 Few-Shot Demos: 0 件 ➔ 4 件 (差分: +4 件)
============================================================
```

## 2. Langfuse によるトレーシング

```mermaid
flowchart TD
    D[DSPy Module / Optimizer] -->|OpenTelemetry OTLP| I[OpenInference DSPy Instrumentor]
    I -->|自動送信| L[Langfuse Dashboard]
```

---

# Chapter 6: 高度な技術深掘り（Assertions, Pydantic, 過学習対策）

## 1. DSPy Assertions と動的バックトラッキング (`dspy.Assert`)

```python
# 制約違反時にLLMにエラー理由を与えて自動再試行（バックトラック）
dspy.Assert(
    len(lines) == 3,
    f"箇条書きはちょうど3点である必要があります（現在 {len(lines)} 行）。修正してください。",
    backtrack=True
)
```

## 2. Pydantic による型安全な構造化抽出

```python
from pydantic import BaseModel, Field
import dspy

class RiskReport(BaseModel):
    is_safe: bool = Field(description="安全かどうか")
    risk_score: float = Field(description="リスクスコア 0.0〜1.0")

class AuditSignature(dspy.Signature):
    contract_text: str = dspy.InputField()
    report: RiskReport = dspy.OutputField()
```

## 3. 過学習（Overfitting）防止策
1. **厳格な 3 分割**: Train (70%), Val (15%), Test (15%)
2. **Temperature 分離**: 指示文生成時は `1.0`（多様性）、推論・評価時は `0.0`（決定性）
3. **Reward Hacking 防止**: Judge に `reasoning` を強制し、冗長さに加点しない

---

# Chapter 7: コミュニティ最新動向 & 次世代技術 (GEPA, ReActV2, BetterTogether)

## 1. GEPA (Genetic-Pareto Evolutionary Prompt Optimization)
遺伝的アルゴリズムと多目的パレート最適化を用い、「**精度 ＋ 応答文字数の短さ ＋ トークンコスト**」の相反する目的を同時に最適化。

## 2. BetterTogether (プロンプト最適化 ＋ LoRA Fine-Tuning)
MIPROv2 で生成した高品質トレースで小規模モデル（Llama等）を LoRA ファインチューニングし、そのモデル向けにプロンプトを再最適化する相互強化ループ。

---

# Practice: ハンズオン実践課題集 & 解答例

### 課題 1: カスタマーサポートの緊急度判定 Signature を作成せよ
- 入力: `customer_message: str`
- 出力: `urgency: str` (High, Medium, Low), `department: str` (Billing, Tech, General)

<details>
<summary><b>課題 1 の解答例</b></summary>

```python
import dspy

class SupportTriageSignature(dspy.Signature):
    """顧客メッセージから緊急度(urgency)と対応部署(department)を判定する。"""
    customer_message: str = dspy.InputField(desc="問い合わせ本文")
    urgency: str = dspy.OutputField(desc="High, Medium, Low のいずれか")
    department: str = dspy.OutputField(desc="Billing, Tech, General のいずれか")
```
</details>

### 課題 2: 部分一致を考慮した損失関数（Metric）を実装せよ
- urgency と department の双方が正解なら 1.0 点
- 片方のみ正解の場合、学習時 (`trace is not None`) は 0.5 点、評価時 (`trace is None`) は 0.0 点

<details>
<summary><b>課題 2 の解答例</b></summary>

```python
def triage_metric(example, pred, trace=None) -> float:
    urgency_match = (getattr(pred, "urgency", "").lower() == getattr(example, "expected_urgency", "").lower())
    dept_match = (getattr(pred, "department", "").lower() == getattr(example, "expected_dept", "").lower())

    if urgency_match and dept_match:
        return 1.0

    if trace is not None:
        return 0.5 if (urgency_match or dept_match) else 0.0

    return 0.0
```
</details>

---

## 🏁 全体まとめ

```mermaid
flowchart LR
    1[1. Signature 定義] --> 2[2. Module 構築]
    2 --> 3[3. Metric & Dataset 作成]
    3 --> 4[4. Optimizer 実行]
    4 --> 5[5. diff 確認 & 評価]
    5 --> 6[6. Langfuse モニタリング]
```

これですべての知識・技術・実践パターンを網羅しました！
