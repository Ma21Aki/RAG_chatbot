# サークル マニュアル RAGシステム

## 概要

サークルの書類・マニュアル（PDF）に対して、自然言語で質問できる RAG（Retrieval-Augmented Generation）システム。ハルシネーションを抑制し、出典を明示します。

## セットアップ

### 1. 環境変数設定

```bash
cp ../.env.example ../.env
```

`.env` に以下を記入：

```
GEMINI_API_KEY=your_api_key_here
PDF_DATA_DIR=./data/pdfs
CHROMA_DB_PATH=./data/chroma_db
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CHROMA_TOP_K=5
GEMINI_MAX_TOKENS=500
GEMINI_TEMPERATURE=0.3
```

### 2. Python 環境セットアップ

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. PDF インデックス構築

```bash
python scripts/rebuild_index.py --pdfs ./data/pdfs --db ./data/chroma_db
```

出力例：
```
📁 PDF ディレクトリ: ./data/pdfs
💾 Chroma DB パス: ./data/chroma_db
🔧 Embedding モデルを初期化中...
🔄 既存インデックスを削除中...
📄 Found 15 PDF files

[1/15] manual.pdf
  ✓ 10 pages processed
...

✓ インデックス完了
  - 処理 PDF: 15 ファイル
  - 総チャンク数: 250
```

## 実験実行

### ターミナル CLI で実験

```bash
python scripts/cli.py
```

対話形式で質問できます：

```
質問: ねっしーとは何ですか？

⏳ 処理中...

【回答】
ねっしーは...

【出典】
  • マニュアル.pdf (p.5)
  • 用語辞典.pdf (p.2)
```

### Jupyter Notebook で段階的にテスト

（準備中 - notebooks/ 以下で段階的なテストを実装予定）

## ファイル構成

```
backend/
  app/
    rag/
      embedder.py      # Embedding（sentence-transformers）
      retriever.py     # ベクトル検索（Chroma）
      generator.py     # 回答生成（Gemini API）
      pipeline.py      # RAG パイプライン全体
      prompt_templates.py
    config.py          # 環境変数管理
    main.py            # FastAPI アプリ（後実装）
  requirements.txt

scripts/
  rebuild_index.py     # PDF インデックス再構築
  cli.py              # ターミナル CLI

notebooks/
  （Jupyter テスト予定）

data/
  pdfs/               # PDF ファイル
  chroma_db/          # ベクトル DB
  backups/            # バックアップ
```

## 技術スタック

- **Embedding**: sentence-transformers (all-MiniLM-L6-v2)
- **ベクトル DB**: Chromadb
- **生成 API**: Gemini API
- **PDF 処理**: PyPDF
- **Web フレームワーク**: FastAPI（後実装）

## 次のステップ

- [ ] CLI での実験・精度検証
- [ ] Jupyter Notebook での段階的テスト
- [ ] FastAPI Web インターフェース実装
- [ ] ラズパイへのデプロイ
- [ ] バックアップ・ロギング機構

## トラブルシューティング

**Q: Embedding モデルがダウンロードできない**
- A: インターネット接続を確認してください。初回は時間がかかります。

**Q: Gemini API のエラー**
- A: `.env` の `GEMINI_API_KEY` が正しいか確認してください。

**Q: Chroma DB が見つからない**
- A: `rebuild_index.py` を実行してインデックスを構築してください。

## ドキュメント

- [要件定義](../docs/requirements.md)
- [技術設計](../docs/technical_design.md)
