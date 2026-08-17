# Chapter 7: コミュニティ最新動向 & 次世代技術 (GEPA, ReActV2, BetterTogether)

DSPy は学術プロジェクトからプロダクション標準フレームワークへと進化を遂げ、2025〜2026 年にかけて数多くの次世代最適化技術が登場しています。

---

## 1. DSPy 3.0+ の Unified API

DSPy 3.x では `dspy.Predict` が統合基底となり、フラグ設定で推論挙動を柔軟に切り替えられるようになりました。

```python
# 1. 単純推論
pred = dspy.Predict(MySignature)

# 2. Chain of Thought (reasoning=True)
cot = dspy.ChainOfThought(MySignature)

# 3. ツール呼び出し付き Agent (Native Tool Calling)
react = dspy.ReAct(MySignature, tools=[search_tool, calculator_tool])
```

- **Provider-Neutral `dspy.LM`**: OpenAI, Anthropic, Gemini, Ollama, Bedrock 等を同一のインターフェースで型安全に呼び出し可能。
- **軽量化**: 余計な依存パッケージが削減され、コンパイル速度が大幅に高速化。

---

## 2. 次世代 Optimizer の台頭

### 1. GEPA (Genetic-Pareto Evolutionary Prompt Optimization)
遺伝的アルゴリズムと多目的パレート最適化を組み合わせた最先端 Optimizer。
- **メリット**: 単一のスコアだけでなく、「**精度 ＋ 応答文字数の短さ ＋ トークンコスト**」という相反する複数の目的関数に対してトレードオフ（パレート解集合）を同時に探索。

```mermaid
flowchart LR
    A[初期プロンプト集団] --> B[交差 & 突然変異<br>指示文の組み合わせ生成]
    B --> C[マルチメトリクス評価<br>・精度<br>・トークン数<br>・レイテンシ]
    C --> D[パレートフロント抽出<br>優秀な個体を選択]
    D -->|次世代へ| B
```

### 2. BetterTogether (プロンプト最適化 ＋ 軽量Fine-Tuning)
プロンプト最適化（MIPROv2等）とモデルの重み更新（LoRA / SFT）を単一のループで相互強化するアプローチ。
1. プロンプト最適化で高品質な推論トレースを生成
2. そのトレースを用いて小規模オープンモデル（Llama, Mistral等）を LoRA で Fine-tune
3. Fine-tune されたモデル向けにプロンプトを再最適化

---

## 3. ReActV2 & DSPy Flex (構造の自動探索)

### ReActV2 (Native Tool Calling)
モダンな LLM の Function Calling / Tool Calling 仕様にネイティブ対応し、並列ツール呼び出し（Parallel Tool Calling）や構造化入出力との親和性が大幅に向上しました。

### DSPy Flex
プロンプトや Few-Shot のテキスト調整にとどまらず、**「単一のCoTで解くべきか、複数のサブタスクに分解してアンサンブル評価すべきか」というプログラムの構造（グラフ）そのものをアルゴリズムが探索・決定** する実験的機能。

---

## 4. 2026年のベストプラクティスまとめ

1. **プロンプトはコードである（Prompts as Code）**: プロンプト文字列を Git で手動管理するのではなく、Signature + Module + Metric + Dataset を管理する。
2. **モデル変更は `re-compile` 1行で完了**: モデル乗り換え（Claude 3.5 ➔ Gemini 3.7 等）の際、プロンプトの調整コストをゼロにする。
3. **Observability の常時稼働**: Langfuse / OpenTelemetry を接続し、プロダクション推論の劣化を検知したら自動で再コンパイルをトリガーする。

---

## 5. 理解度チェック & ミニクイズ

<details>
<summary><b>Q1. GEPA は MIPROv2 と比較してどのようなユースケースで特に威力を発揮しますか？</b></summary>

**A1.** 「精度を上げたいが、プロンプトが長くなりすぎてコストやレイテンシが悪化するのは困る」といった、**精度・コスト・レイテンシの複数トレードオフ（多目的最適化）** を同時に満たすプロンプトを見つけたい場合に非常に有効です。
</details>

<details>
<summary><b>Q2. BetterTogether アプローチの最大の利点は何ですか？</b></summary>

**A2.** プロンプト最適化だけでは限界がある小規模・安価なローカルモデル（7B〜14Bクラス）に対して、プロンプト最適化で得られた高品質トレースを使って蒸留・LoRA学習を行い、GPT-4o クラスの性能を低コスト・高スループットで実現できる点です。
</details>

---

- 前の章: [Chapter 6: 高度な技術深掘り (06_deep_dive_assertions_pydantic.md)](06_deep_dive_assertions_pydantic.md)
- 目次へ戻る: [Index](../INDEX.md)
