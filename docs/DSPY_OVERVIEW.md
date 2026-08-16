# DSPy 概要 & キャッチアップガイド

## 1. DSPy とは何か？

**DSPy (Declarative Self-improving Python)** は、LLM（大規模言語モデル）へのプロンプトやパイプラインを**手作業の文字列調整から、プログラムによる構造化と自動最適化へシフトさせるフレームワーク**です。

### 「LLMのチューニングを自動でやる」とはどういうことか？

一般的な「LLMのチューニング」には以下のような種類があります：
1. **ファインチューニング (Fine-Tuning)**: LLMの「重み（Weight）」自体を更新する。
2. **プロンプトチューニング / 最適化 (Prompt Optimization)**: LLMへ送る「プロンプト（指示文 + Few-Shot例）」を自動生成・選択して最適化する。

**DSPy は主に 2（プロンプト最適化）を自動化するツール**です（一部、モデル重みのファインチューニングを行うOptimizerも提供されています）。

---

## 2. 従来のプロンプトエンジニアリングとの違い

| 項目 | 従来のプロンプトエンジニアリング | DSPy によるアプローチ |
| :--- | :--- | :--- |
| **開発スタイル** | プロンプトの文字列を手作業で試行錯誤 (Trial & Error) | タスクの**入出力構造（Signature）**と**ロジック（Module）**をコードで定義 |
| **最適化手法** | 人間の勘・感覚で修正 | 評価関数（Metric）に基づき、**Optimizer（アルゴリズム）が自動最適化** |
| **Few-Shot（例）** | 人間が手動で良い例を選んで埋め込む | 評価スコアの高い実行履歴から**最適なFew-Shot例を自動抽出** |
| **モデル変更対応** | モデルを変えるたびにプロンプトを全面書き直し | コードはそのまま、**別モデル用にOptimizerを再実行**するだけ |

---

## 3. DSPyの4大核心概念（PyTorchとの対比）

DSPy は **PyTorch に非常に近いメンタルモデル**で設計されています。

```
PyTorch:  データ + モデル構造(nn.Module) + 損失関数(Loss) + 最適化(Optimizer) -> 重みの最適化
DSPy:     データ + パイプライン(dspy.Module) + 評価関数(Metric) + 最適化(Optimizer) -> プロンプト/Few-shotの自動生成
```

1. **Signatures (宣言的インターフェース)**
   - 「何を入力して、何を出すか」の型定義。文字列ではなく `question -> answer` や `context, question -> answer: float` のように記述。
2. **Modules (処理コンポーネント)**
   - `dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct` など、LLMにどのように考えさせるかのビルディングブロック。
3. **Metrics (評価関数)**
   - 出力結果の良し悪しを定義する関数（0.0〜1.0 のスコアや判定結果を返す）。LLM-as-a-Judge も組み込み可能。
4. **Optimizers / Teleprompters (自動最適化アルゴリズム)**
   - `BootstrapFewShot`: 成功した実行例をFew-Shotとしてプロンプトに集める。
   - `MIPROv2`: 指示文（Instruction）とFew-Shot例を同時に最適化する。

---

## 4. なぜ今 DSPy なのか？（導入メリット）

- **プロンプト沼からの脱却**: 「語尾をどうするか」「どんな条件を足すか」で悩む時間をゼロに。
- **再現性と評価基盤の確立**: テストデータとMetricを定義しないと最適化できないため、自然と「評価駆動開発 (Eval-Driven Development)」が身につく。
- **モデル非依存性**: Claude, GPT-4o, Gemini, ローカルLLM(Ollama等) へ変更した際も、最適化を実行し直すだけで各モデルに最適なプロンプトが自動生成される。

---

## 5. DSPyで作成するプロンプト最適化ループ（LaaJ-dspy-looper）

本プロジェクト（`LaaJ-dspy-looper`）では、社内の **LLM-as-a-Judge (LaaJ)** と **DSPy** を統合し：

1. **テストデータ投入**
2. **DSPy Moduleによる推論**
3. **LLM-as-a-Judge / カスタムMetricによる自動スコアリング**
4. **DSPy Optimizerによるプロンプト自動改善**

という完全に閉じた**自動最適化ループ**を構築します。
