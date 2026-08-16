# DSPy コミュニティ最新動向 & トレンド (2025–2026)

DSPy は 2024 年の学術研究プロジェクトから急速に進化し、2025〜2026 年にかけてプロダクション開発で広く利用される「宣言的 LLM プログラミング」の標準フレームワークへと成長しました。

---

## 1. 主要アップデートと進化の歴史

### DSPy 3.0 / 3.3 のリリース（プロダクション対応の成熟）
2025年中盤の **DSPy 3.0**、およびそれに続く **DSPy 3.3.0** のリリースにより、フレームワークの抽象化とAPIが大幅に統一・安定化しました。

- **Unified API**: `dspy.Predict` が統合され、設定フラグ（`reasoning=True` で CoT、`tools=[...]` で ReAct）によって推論挙動をシームレスに変更可能になりました。
- **Provider-Neutral `BaseLM`**: LiteLLM や 各種ベンダーAPI（OpenAI, Anthropic, Gemini, Ollama 等）との接続が型安全かつ統一的に整理されました。
- **軽量化・依存関係の整理**: 不要なサードパーティ依存を削減し、高速なコンパイルと省メモリ化を実現。

---

## 2. 次世代 Optimizer（自動最適化アルゴリズム）の台頭

### GEPA (Genetic-Pareto Evolutionary Prompt Optimization)
2025後半〜2026年にかけて登場した最新の最適化アルゴリズム。
- **特徴**: 遺伝的アルゴリズムとパレート最適化を組み合わせ、指示文（Instruction）とFew-Shot例を同時に進化させます。
- **優位性**: 従来の強化学習（RL）ベースのプロンプト最適化よりも少ない計算コストで、マルチメトリクス（例: 精度 + 応答の短さ + コスト）に対する最適なプロンプト集合を発見します。

### MIPROv2 (Multiprompt Instruction PRoposal Optimizer v2)
現在もプロダクションにおける**標準Optimizer**として君臨しています。
- Meta-LLM による指示文候補の多様な提案（Proposal）
- Optuna の TPE (Tree-structured Parzen Estimator) によるベイズ最適化
- ミニバッチ評価による高速な収束

### BetterTogether
プロンプト最適化（MIPROv2等）とモデルの軽量ファインチューニング（LoRA / SFT）を**単一のコンパイルループで統合**するアプローチ。
- プロンプト最適化で良いトレースを生成 ➔ そのトレースでモデルをFine-tune ➔ Fine-tuneされたモデル用にプロンプトを再最適化、という相互強化ループを実現。

---

## 3. エージェント機能の強化（ReActV2 & DSPy Flex）

### ReActV2 と ネイティブ Tool-Calling
従来のマニュアルテキストベースの ReAct から、モダンなツール呼び出し（Native Tool Calling）仕様へ移行しました。
- `dspy.History`, `dspy.Tool`, `dspy.ToolCalls` などの型定義を導入。
- **並列ツール呼び出し (Parallel Tool Calls)** をサポートし、構造化データ・型定義（Pydantic）との親和性が飛躍的に向上。

### DSPy Flex (構造の自動探索)
プロンプトや Few-Shot のテキスト調整にとどまらず、**プログラムのパイプライン構造（タスクの分解グラフ）そのものを探索・最適化**する実験的機能。
- 例: 単一のCoTで解くべきか、複数のサブタスクに分解してアンサンブル評価すべきかをアルゴリズムが決定。

---

## 4. コミュニティのベストプラクティスと評価（Eval-Driven）

1. **Prompt Decay（プロンプト劣化）の回避**:
   - モデルを切り替えた際（例: GPT-4o ➔ Llama-3 ➔ Gemini 3.7）、プロンプトを手動修正するのではなく「DSPy コンパイルを再実行する」ことが業界標準に。
2. **LLM-as-a-Judge の堅牢化**:
   - メトリクス（Metric）に対する**Reward Hacking（報酬ハック）**を防ぐため、Judgeのドキュメントに負の基準を明記し、`reasoning` フィールドを必須化することが推奨される。
3. **Langfuse / OpenTelemetry によるトレーシング**:
   - コンパイル中の試行錯誤トレースやプロダクション推論の状況を高度に可視化。
