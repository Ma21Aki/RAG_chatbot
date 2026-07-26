# サークル マニュアル RAGシステム（Python版）

## 概要

サークルの書類・マニュアル（PDF）に対して、自然言語で質問できる RAG（Retrieval-Augmented Generation）システム。ハルシネーションを抑制し、出典を明示します。

この Python版は精度改良の実験場です。実運用（ラズパイ常時稼働＋ブラウザUI）は JS版
[nec_chatbot_js](../nec_chatbot_js/README.md) を使います。プロジェクト全体の位置づけは
[ルートのREADME](../README.md) を参照。

検索の流れ:

```
質問 → 用語注入（辞書の定義を連結）
     → ベクトル検索（Chroma） ＋ BM25検索（Janome分かち書き）
     → RRFで統合 → 係名がファイル名に一致したらブースト
     → 上位チャンクを Gemini に渡して回答生成
```

## セットアップ

初めて環境を作る場合は、画面つきの詳しい手順書 [SETUP.md](../SETUP.md) の方が親切です。
以下は要点のみ。

### 1. 環境変数設定

`.env` はプロジェクトルート（`nec_chatbot` の一つ上）に置きます。

```bash
cp ../.env.example ../.env
```

`.env` の主な項目（値は `.env.example` のままで動きます）:

```
GEMINI_API_KEY=your_api_key_here
PDF_DATA_DIR=./data/pdfs
CHROMA_DB_PATH=./data/chroma_db
EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
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

埋め込みモデル（約470MB）は初回実行時に自動ダウンロードされます。

### 3. PDF インデックス構築

管理者から `chroma_db` を受け取っている場合は不要です。文書を追加・更新したときだけ実行します。

```bash
python scripts/rebuild_index.py
```

パスやモデルを変えたいときは `--pdfs` / `--db` / `--model` で上書きできます（既定値は `.env`）。

出力例：
```
[DIR] PDF directory: ./data/pdfs
[DB] Chroma DB path: ./data/chroma_db

[INIT] Initializing Embedding model...

[1/15] manual.pdf
  [OK] 10 pages processed
...

==================================================
[DONE] Index construction completed
  - PDF files processed: 15
  - Total chunks: 250
==================================================
```

索引構築時の扱い:

- PDFはページ単位で抽出し、500字・オーバーラップ50字でチャンク化
- Markdown は見出し（# 行）単位で分割。用語辞典のような「見出し＝1項目」の文書向け
- 画像PDFは同じ場所に同名の `.md` を置くと、その内容に差し替わる（OCR起こし）
- 空白圧縮後30字未満のチャンクは、更新履歴や空ページとみなして除外

## 実験実行

### Jupyter Notebook（おすすめ）

```bash
jupyter notebook scripts/ipynb/experiment.ipynb
```

VSCode で開き、カーネルに `backend/venv` を選ぶ方法でも可。上のセルから順に実行し、
質問セルで次を呼びます。

```python
ask_to_chatbot("鉈を使うときの注意点は？")  # 回答＋出典＋処理時間＋トークン数
serche_query("肝試し")                      # 検索のみ（Gemini不要）
```

セル構成や出力の見方は [scripts/ipynb/README.md](scripts/ipynb/README.md) に詳しくあります。

### ターミナル CLI

```bash
python scripts/cli.py
```

対話形式で質問できます（`exit` で終了）：

```
質問: ねっしーとは何ですか？

⏳ 処理中...

【回答】
ねっしーは...

【出典】
  • マニュアル.pdf (p.5)
  • 用語辞典.pdf (p.2)
```

### FastAPI

`backend/app/main.py` に `/health` と `/query` の最小実装があります（ポート8000）。
現状は動作確認用で、Web UI は JS版を使う方針です。

## 検索まわりの調整

内輪語対策のデータは以下で管理します。文書を足したら、あわせて見直します。

| ファイル | 役割 |
|---|---|
| [user_dict.csv](backend/app/rag/user_dict.csv) | 分かち書きのユーザー辞書。総リー・IB 等が誤分割されるのを防ぐ |
| [kakari_terms.csv](backend/app/rag/kakari_terms.csv) | 係名リスト。質問中の係名がファイル名に一致したら加点する |
| `data/pdfs/` 配下の `*用語辞典*.md` | 用語注入の定義元。`##` 見出しが用語、別名は `・` や `、` 区切り |

主な調整パラメータ:

| 場所 | パラメータ | 意味 | 既定 |
|---|---|---|---|
| `.env` | `CHROMA_TOP_K` | 生成に渡す検索結果の件数 | 5 |
| `.env` | `GEMINI_MAX_TOKENS` | 生成の最大トークン数 | 500 |
| `.env` | `GEMINI_TEMPERATURE` | 温度（0〜2、低いほど確定的） | 0.3 |
| [hybrid_retriever.py](backend/app/rag/hybrid_retriever.py) | `candidate_k` | 各検索から集める候補数 | 30 |
| [hybrid_retriever.py](backend/app/rag/hybrid_retriever.py) | `rrf_k` | RRF定数。大きいほど順位差が緩やか | 60 |
| [hybrid_retriever.py](backend/app/rag/hybrid_retriever.py) | `boost_weight` | 係名一致の加点量 | 0.03 |
| [rebuild_index.py](scripts/rebuild_index.py) | `chunk_size` / `overlap` | チャンク幅・重なり | 500 / 50 |

## スクリプト

| スクリプト | 用途 |
|---|---|
| [rebuild_index.py](scripts/rebuild_index.py) | PDF/MD → チャンク分割 → 索引構築 |
| [evaluate.py](scripts/evaluate.py) | ベクトル / BM25 / hybrid の精度比較（Hit@k・MRR）。評価データは [eval_set.json](scripts/eval_set.json) |
| [try_hybrid.py](scripts/try_hybrid.py) | 1問について3方式の上位結果を並べて目視比較 |
| [dump_chunks.py](scripts/dump_chunks.py) | 索引済みチャンクをCSVへ出力。中身の薄い壊れチャンク探し用 |
| [cleanup_orphaned_vectors.py](scripts/cleanup_orphaned_vectors.py) | 削除済みファイルのベクトルを索引から除去 |
| [cli.py](scripts/cli.py) | ターミナルでの対話実験 |
| [test_folder_metadata.py](scripts/test_folder_metadata.py) | フォルダメタデータ付き検索の動作確認 |

## ファイル構成

```
backend/
  app/
    rag/
      embedder.py           # 埋め込み（sentence-transformers）
      retriever.py          # ベクトル検索（Chroma）
      bm25_retriever.py     # 語彙検索（BM25）
      hybrid_retriever.py   # 用語注入＋RRF統合＋係名ブースト
      tokenizer.py          # 分かち書き（Janome＋user_dict.csv）
      glossary.py           # 用語辞典からクエリへ定義を注入
      generator.py          # 回答生成（Gemini API / gemini-2.5-flash）
      pipeline.py           # RAG パイプライン全体
      prompt_templates.py   # システムプロンプト・出典整形
      user_dict.csv         # 内輪語のユーザー辞書
      kakari_terms.csv      # 係名リスト
    config.py               # 環境変数管理
    main.py                 # FastAPI（最小実装）
  requirements.txt

scripts/
  rebuild_index.py, evaluate.py, try_hybrid.py, dump_chunks.py,
  cleanup_orphaned_vectors.py, cli.py, test_folder_metadata.py
  ipynb/
    experiment.ipynb        # 実験の主入口
    README.md

data/                       # git管理外（管理者から受け取る）
  pdfs/                     # PDF・OCR起こしのMD
  chroma_db/                # ベクトルDB
  backups/
```

## 技術スタック

- Embedding: sentence-transformers（paraphrase-multilingual-MiniLM-L12-v2 / ローカル・無料）
- ベクトル DB: Chroma
- 語彙検索: BM25（分かち書きは Janome）
- 生成 API: Gemini API（gemini-2.5-flash / google-genai）
- PDF 処理: pypdf

## 次のステップ

- [x] CLI・Notebook での実験環境
- [x] ハイブリッド検索（BM25 ＋ RRF）と内輪語の辞書対応
- [x] 検索精度の定量評価（evaluate.py）
- [ ] チャンキング方法の見直し（parent-child / context-expansion）
- [ ] 使用文書の選定と構造化（暗黙知の文書化）
- [ ] HyDE（仮想的回答に類似するチャンクも探す）

運用面（Web UI・ラズパイ常時稼働）は JS版で進めています。

## トラブルシューティング

Q: Embedding モデルがダウンロードできない
- A: インターネット接続を確認してください。初回は数分かかります。

Q: Gemini API のエラー（429 RESOURCE_EXHAUSTED 等）
- A: `.env` の `GEMINI_API_KEY` と、AI Studio の請求設定を確認。検索だけなら `serche_query()` は Gemini 不要で動きます。

Q: Chroma DB が見つからない
- A: `data/chroma_db/` にデータを配置するか、`rebuild_index.py` を実行してください。

Q: `ModuleNotFoundError: No module named 'app.rag'`
- A: venv を有効化しているか、`pip install -r requirements.txt` 済みかを確認。

Q: 索引構築が MemoryError で落ちる
- A: 巨大PDFのページ抽出でメモリを使い切っています。そのPDFを分割するか、他アプリを閉じて再実行してください。

## ドキュメント

- [要件定義](../docs/requirements.md)
- [技術設計](../docs/technical_design.md)
- [データ変更履歴](docs/data_changes.md) … PDFの改名・OCR差し替えの記録（PDFはgit管理外のため）
- [Gemini API ガイド](docs/gemini_api_guide.md)
