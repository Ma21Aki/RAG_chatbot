# サークル マニュアル RAGシステム

サークルの書類・マニュアル（PDF）に自然言語で質問できる RAG システム。
回答には出典（ファイル名・ページ）を明示し、ハルシネーションを抑制します。

処理の流れ: 質問 → ベクトル化(embed) → Chromaで類似チャンク検索 → Geminiが回答生成

---

## まず見る

- セットアップして動かす → [SETUP.md](SETUP.md)
- 実験する（質問を投げる入口） → [experiment.ipynb](nec_chatbot/scripts/ipynb/experiment.ipynb)
  - `ask_to_chatbot("質問")` … 回答＋出典＋メトリクス
  - `serche_query("質問")` … 検索のみ（Gemini不要）

## 主要ファイル

| パス | 役割 |
|---|---|
| [pipeline.py](nec_chatbot/backend/app/rag/pipeline.py) | RAG全体の流れ（まずこれを読むと全体像がわかる） |
| [embedder.py](nec_chatbot/backend/app/rag/embedder.py) | テキストのベクトル化（sentence-transformers） |
| [retriever.py](nec_chatbot/backend/app/rag/retriever.py) | Chromaでのベクトル検索 |
| [generator.py](nec_chatbot/backend/app/rag/generator.py) | Geminiでの回答生成 |
| [prompt_templates.py](nec_chatbot/backend/app/rag/prompt_templates.py) | システムプロンプト・出典整形 |
| [config.py](nec_chatbot/backend/app/config.py) | 設定の読み込み（.envを参照） |
| [rebuild_index.py](nec_chatbot/scripts/rebuild_index.py) | PDF→チャンク分割→索引構築 |
| [.env.example](.env.example) | 環境変数のひな形（コピーして`.env`を作る） |

詳細な手順や構成は [nec_chatbot/README.md](nec_chatbot/README.md) を参照。
要件・設計は [requirements.md](docs/requirements.md) / [technical_design.md](docs/technical_design.md)。

## 技術スタック

- 埋め込み: sentence-transformers（paraphrase-multilingual-MiniLM-L12-v2 / ローカル・無料）
- ベクトルDB: Chroma
- 生成: Gemini API
- PDF処理: pypdf

---

## 現在のできること

- 質問文とpdfの500字毎ブロックを類似度比較し、それをAIに投げて回答生成。参照元も提示
- 直接的言及があればそこそこの精度で拾える。計算も重くない
- 1回のチャット（検索上位5件を参照）で
  - 約2,000トークン消費（大半が検索結果の読み込み）
  - 所要時間 3秒前後
- 初回設定: PDFをチャンク分割・ベクトル化して索引を構築。一度設定したら永続（[rebuild_index.py](nec_chatbot/scripts/rebuild_index.py)）


## できないこと

- 文中で直接言及されていない問いには答えにくい
  - 例: 「総リーと副リーの仕事の違いは？」（比較・推論が必要な問い）
- 同一pdfの中でも距離が離れると検出力が落ちる

## 今後の改良案

- tier1（すぐやる）: 
    - ★シンプルな単語検索ベースの検索 ⇨ 固有名詞に強くなる
    - ★チャンキング方法の見直し ⇨ メタタグ設定やオーバーチャンキング幅等、検索後に何をAIに渡すかの選定。具体的にはparent-childやcontext-expantion。クエリ消費がきついのがネックだが
- tier2（やりたい）:
    - 使用文書の選定と構造化 ⇨ 現状は暗黙知を含むねっしー基本情報が足りない。 
    - HyDE ⇨　質問に類似するチャンクではなく、質問への仮想的回答に類似するチャンクも探させる
- tier3（後で）:
    - ★ハイブリッド検索 ⇨　ベクトル変換のモデルを複数組み合わせる。具体的にはBM25(TFIDF的)を用いる。単語検索の上位互換。初回計算が重いが最終的にはやりたい。
    - embeddingモデルの変更 ⇨ 現状のminiLMを日本語特化の ruri v3に変える等。現状そこまで困ってなさげ
