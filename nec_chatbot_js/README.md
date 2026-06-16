# nec_chatbot_js

サークル書類・マニュアル RAGシステムの JavaScript（Node.js）版。
検索・埋め込みはローカル完結、生成のみ Gemini API。ラズパイ常時稼働＋ブラウザUIを想定。

設計: [docs/requirements.md](docs/requirements.md) / [docs/technical_design.md](docs/technical_design.md)

## セットアップ

```
npm install
cp .env.example .env     # GEMINI_API_KEY を記入（生成に必要）
```

## 索引構築（文書を追加・更新したら実行）

```
npm run build-index
```
data/docs 配下の PDF/MD を取り込み、ローカル埋め込みして data/index.json を作る。
画像PDFは同じ場所に同名の .md を置くと、その内容に差し替わる（OCR起こし）。

## 検索・回答の確認（CLI）

```
node scripts/search.js "総リーの仕事は"          # 検索のみ（LLM不要）
node scripts/search.js --ask "総リーの仕事は"     # 生成まで（要 GEMINI_API_KEY）
```

## サーバー起動

```
npm start                # http://localhost:3000
```
ブラウザでアクセスすると質問でき、回答と出典が表示される。

- POST /api/ask      `{ "query": "..." }` → 回答＋出典＋メトリクス
- POST /api/search   `{ "query": "..." }` → 検索結果のみ（LLM不要）
- GET  /api/stats    → 今月の質問回数・入出力トークン総量（UIの「設定」から表示）
- POST /api/feedback `{ "type", "detail", "query", "answer" }` → 不具合報告を記録

## 利用統計・不具合報告

- 検索履歴: ブラウザの localStorage に直近10件を保存（入力欄クリックで表示）。サーバーには送らない。
- 利用統計: 質問のたびに `data/stats.json` へ月別集計（質問回数・入出力トークン）。UI左上の「設定」から閲覧。
- 不具合報告: 回答下部のフォームから送信され `data/feedback.jsonl` に1行1件で追記される。
  管理者はこのファイルを直接確認して対応する（メール等の通知はしない）。
  どちらも実行時に生成される（gitignore対象）。

## 構成（Python版との対応）

| 機能 | ファイル |
|------|----------|
| 索引構築 | scripts/buildIndex.js |
| 埋め込み | src/rag/embedder.js（transformers.js） |
| ベクトル検索 | src/rag/vectorStore.js（総当りcosine） |
| 分かち書き | src/rag/tokenizer.js（kuromoji + 用語保護層） |
| BM25 | src/rag/bm25.js |
| 用語注入 | src/rag/glossary.js |
| RRF統合＋係名ブースト | src/rag/hybrid.js |
| 文脈整形 | src/rag/contextFormatter.js |
| 生成（Gemini・差し替え可能） | src/rag/generator/ |
| パイプライン | src/rag/pipeline.js |
| サーバー | src/server/、src/index.js |
| UI | public/（Alpine.js） |

## 注意

- 64bit の Raspberry Pi OS / Node.js LTS で動作（onnxruntime-node のARM64対応のため）。
- 索引は JS 側の埋め込みで作る。Python版の Chroma 索引は流用不可。
- 外部公開する場合は .env の AUTH_TOKEN を設定し、TLS は Tunnel（Tailscale Funnel / Cloudflare Tunnel）で終端する。
- Windows のシェルから curl に日本語を直接渡すと文字化けすることがある。ブラウザUIや Node からのリクエストは UTF-8 で正しく届く。
