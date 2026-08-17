# LaaJ-dspy-looper ドキュメント & 学習コンテンツ

本プロジェクト (`LaaJ-dspy-looper`) のドキュメントおよび学習コンテンツ一覧です。

---

## 📚 体系的学習ガイド（チュートリアル・全7章 完全版）

スマートフォンや外出先の GitHub 上で快適に学習できるステップバイステップのチュートリアルです。

- **[全章統合マスターガイド (LEARNING_GUIDE.md)](LEARNING_GUIDE.md)**: 全7章 ＋ 実践課題集を一気通読できる完全版ガイド
- **[ブラウザ用リッチHTMLビューアー (index.html)](index.html)**: スマホ・PCのブラウザで読めるインタラクティブHTML

### 各章個別ドキュメント
1. **[Chapter 1: DSPy メンタルモデル & 思考のパラダイムシフト](tutorials/01_mental_model.md)**（Prompt Decay 回避、開発者とアルゴリズムの黄金律）
2. **[Chapter 2: 入出力設計（Signature）& モジュール構築（Module）](tutorials/02_signature_and_modules.md)**（`Predict` vs `ChainOfThought`、`inspect_history`）
3. **[Chapter 3: 評価関数（Metric）エンジニアリング & trace引数](tutorials/03_metric_engineering.md)**（損失関数としての連続スコア、Reward Hacking対策）
4. **[Chapter 4: 自動最適化（Optimizer）の実践](tutorials/04_optimizers_in_depth.md)**（`BootstrapFewShot`、`MIPROv2` ベイズ最適化）
5. **[Chapter 5: LLM-as-a-Judge, 差分比較 & Observability](tutorials/05_llm_as_judge_and_tracing.md)**（`laaj diff`、Langfuse トレーシング）
6. **[Chapter 6: 高度な技術深掘り](tutorials/06_deep_dive_assertions_pydantic.md)**（`dspy.Assert` バックトラッキング、Pydantic 型安全抽出、過学習対策）
7. **[Chapter 7: コミュニティ最新動向 & 次世代技術](tutorials/07_community_trends_gepa.md)**（DSPy 3.3, GEPA 多目的パレート最適化, BetterTogether, ReActV2）

---

## 📖 仕様書・技術リファレンス

- **[SPEC.md](SPEC.md)**: プロジェクト全体仕様書・国内外事例調査（LayerX, note, Jina AI 等）
- **[DSPY_OVERVIEW.md](DSPY_OVERVIEW.md)**: DSPy 基本概念・従来手法との違い
- **[DSPY_DEEP_DIVE.md](DSPY_DEEP_DIVE.md)**: 高度な技術深掘りガイド（内部メカニズム, Assertions, Pydantic, MIPROv2, 過学習対策）
- **[DSPY_COMMUNITY_TRENDS.md](DSPY_COMMUNITY_TRENDS.md)**: コミュニティ最新動向 & トレンド（DSPy 3.0+, GEPA, ReActV2, DSPy Flex）
- **[ENABLEMENT_PLAN.md](ENABLEMENT_PLAN.md)**: チームイネーブルメント計画・ロードマップ
