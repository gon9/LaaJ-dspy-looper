# Chapter 6: 高度な技術深掘り（Assertions, Pydantic, 過学習対策）

本章では、プロダクション品質の堅牢な LLM アプリケーションを構築するための高度な技術要素（型安全、自己修復ループ、過学習防止）を学びます。

---

## 1. DSPy Assertions と動的バックトラッキング (`dspy.Assert`)

LLM の出力が期待する制約（例: 「50文字以内」「指定フォーマットに準拠」「引用元が含まれる」）を満たさない場合、通常はエラーで終了するか後処理で捨てるしかありません。
DSPy の **`dspy.Assert`** は、制約違反を検知した瞬間に **LLM へ自動でエラーメッセージをフィードバックし、直前の思考を修正して再生成させる（バックトラッキング）** 仕組みを提供します。

```mermaid
flowchart TD
    A[入力プロンプト] --> B[LLM が回答を生成]
    B --> C{dspy.Assert 条件チェック}
    C -->|合格| D[正常出力]
    C -->|不合格 (制約違反)| E[エラー内容をプロンプトに追加してバックトラック]
    E -->|再試行 (最大N回)| B
```

### 実装例:

```python
import dspy

class StrictOutputSignature(dspy.Signature):
    """入力文を解析し、3点箇条書きかつ各行30文字以内で出力する。"""
    text: str = dspy.InputField()
    summary: str = dspy.OutputField(desc="3点箇条書き")

class RobustSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(StrictOutputSignature)

    def forward(self, text: str):
        pred = self.prog(text=text)
        lines = [line.strip() for line in pred.summary.strip().split("\n") if line.strip()]

        # 制約 1: 箇条書きがちょうど3点であること
        dspy.Assert(
            len(lines) == 3,
            f"箇条書きはちょうど3点である必要があります（現在 {len(lines)} 行）。再出力してください。",
            backtrack=True
        )

        # 制約 2: 各行が30文字以内であること
        for i, line in enumerate(lines, 1):
            dspy.Assert(
                len(line) <= 30,
                f"行 {i} の文字数が長すぎます（{len(line)}文字）。30文字以内に短縮してください。",
                backtrack=True
            )

        return pred
```

> [!NOTE]
> `dspy.Assert` は例外を投げて中断するのではなく、推論トレースを保持したまま LLM に反省文を与えて再試行させます。

---

## 2. Pydantic スキーマによる型安全な構造化抽出

DSPy は Pydantic の BaseModel とシームレスに統合できます。

```python
from pydantic import BaseModel, Field
import dspy

class RiskItem(BaseModel):
    category: str = Field(description="リスクのカテゴリ（法的、財務、技術的等）")
    severity: int = Field(description="重要度 1〜5")
    description: str = Field(description="リスクの詳細説明")

class ContractAnalysisResult(BaseModel):
    is_safe: bool = Field(description="契約書全体が安全かどうか")
    risks: list[RiskItem] = Field(description="検出されたリスク項目のリスト")

class ContractAuditor(dspy.Signature):
    """契約書テキストを監査し、構造化されたリスク分析結果を出力する。"""
    contract_text: str = dspy.InputField()
    audit_report: ContractAnalysisResult = dspy.OutputField()
```

- JSON パースエラーやスキーマ不一致は DSPy が内部で自動修復します。
- IDE の型補完や静的型チェック（mypy, pyright）が完全に効くため、保守性が大幅に向上します。

---

## 3. プロンプト最適化における過学習（Overfitting）とその対策

プロンプト最適化は、重みのファインチューニングと同様に **過学習** が発生します。

### 過学習の兆候:
- Train データの精度は 95% なのに、Test データでは 60% に急落する。
- 最適化された指示文が「特定のキーワード（Trainデータ固有の単語）が含まれる場合は...」と個別事情に過剰適合している。

### 過学習防止のベストプラクティス:

| 対策手法 | 具体的な実装方針 |
| :--- | :--- |
| **1. 厳格なデータセット 3 分割** | `Train` (最適化用: 60〜70%), `Validation` (探索時の評価用: 15〜20%), `Test` (最終評価用: 15〜20%) に分離する。 |
| **2. Temperature の制御** | Meta-LLM の指示文生成時は `temperature: 1.0` で多様性を確保し、評価・推論時は `temperature: 0.0〜0.3` で安定化させる。 |
| **3. 指示文候補数と試行回数の制限** | データ量が少ない（<50件）場合は `MIPROv2` の `num_candidates=3〜5`, `num_trials=5〜10` 程度に抑える。 |
| **4. Metric の報酬ハック防止** | `reasoning` フィールドを必須化し、「文字数が多いだけで高得点」といった抜け穴をなくす。 |

---

## 4. 理解度チェック & ミニクイズ

<details>
<summary><b>Q1. 従来の Python の <code>assert</code> 文と <code>dspy.Assert</code> の根本的な違いは何ですか？</b></summary>

**A1.** 通常の `assert` は条件が False だとプログラムが例外（AssertionError）を投げて停止します。一方 `dspy.Assert` は、エラーメッセージをプロンプトのコンテキストに追加し、**LLM が直前の出力を修正して再生成するリトライループ（バックトラッキング）を自動実行** します。
</details>

<details>
<summary><b>Q2. 訓練データが 20 件しかない状態で MIPROv2 を試行回数 100 回（num_trials=100）回すと何が起こりやすいですか？</b></summary>

**A2.** 20件の訓練データだけに過剰適合した「歪んだ指示文」が選定され、過学習（Overfitting）により汎化性能（未知の入力に対する精度）が大幅に低下する可能性が高くなります。
</details>

---

- 前の章: [Chapter 5: LLM-as-a-Judge & 可視化 (05_llm_as_judge_and_tracing.md)](05_llm_as_judge_and_tracing.md)
- 次の章: [Chapter 7: コミュニティ最新動向 & トレンド (07_community_trends_gepa.md)](07_community_trends_gepa.md)
