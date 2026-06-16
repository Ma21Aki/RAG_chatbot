# RAGシステム（JavaScript版）技術設計書

作成日: 2026-06-16
バージョン: 1.0
対象: nec_chatbot_js（Python版 nec_chatbot の移植）

関連: [requirements.md](./requirements.md)

---

## 1. 設計方針

- Node 単一ランタイム。ビルド工程を持たない（フロントは Alpine.js を静的配信）。
- 検索・埋め込みはローカル完結。生成のみ Gemini API。
- 各処理は小さなモジュールに分割し、Python版の責務に1対1で対応させる。
- 生成は Generator 抽象で差し替え可能にし、将来ローカルLLM/抽出型へ移れる余地を残す。
- ベクトルDBは持たず、索引は単一の JSON ファイル（数百チャンク規模のため総当りで十分）。

---

## 2. アーキテクチャ全体図

```
┌─────────────────────────────────────────────┐
│  利用者ブラウザ（学内外）                       │
│  Alpine.js UI（public/）                       │
└───────────────┬─────────────────────────────┘
                │ HTTPS(Tunnel) / JSON
┌───────────────▼─────────────────────────────┐
│  ラズパイ Node サーバー（Express）              │
│  ┌─ server: ルート・認証・レート制限 ─────┐    │
│  │   POST /api/ask   POST /api/search    │    │
│  └──────────────┬────────────────────────┘    │
│  ┌──────────────▼─ rag/pipeline ──────────┐    │
│  │  glossary(用語注入) → embedder/tokenizer │    │
│  │  → vectorStore + bm25 → hybrid(RRF+boost)│   │
│  │  → contextFormatter → generator(Gemini)  │   │
│  └──────────────┬───────────────────────── ┘   │
│  ┌──────────────▼── ローカルデータ ─────────┐   │
│  │  data/index.json（チャンク＋埋め込み）     │   │
│  │  data/docs/（PDF・MD）  data/dict/（辞書） │   │
│  │  models/（MiniLM ONNX）                   │   │
│  └──────────────────────────────────────────┘   │
└───────────────┬─────────────────────────────┘
                │ HTTPS（生成のみ）
        ┌───────▼────────┐
        │ Gemini API     │
        └────────────────┘
```

---

## 3. ディレクトリ構成

```
nec_chatbot_js/
├─ docs/
│  ├─ requirements.md
│  └─ technical_design.md
├─ data/
│  ├─ docs/                  文書（PDF/MD）の複製
│  ├─ index.json             索引（チャンク＋埋め込み）
│  └─ dict/
│     ├─ kakari_terms.txt    係名リスト（ブースト用）
│     ├─ protect_terms.txt   用語保護層の語彙（1語扱いを強制）
│     └─ glossary.md         用語辞典（用語注入用・複製）
├─ models/                   MiniLM ONNX（ローカル配置）
├─ src/
│  ├─ config.js              設定（.env 読み込み）
│  ├─ rag/
│  │  ├─ embedder.js         transformers.js でベクトル化
│  │  ├─ vectorStore.js      index.json 読み込み＋総当りcosine
│  │  ├─ tokenizer.js        kuromoji + 用語保護層
│  │  ├─ bm25.js             BM25
│  │  ├─ glossary.js         用語注入（クエリ拡張）
│  │  ├─ hybrid.js           RRF統合＋係名ブースト
│  │  ├─ contextFormatter.js 文脈整形①②③
│  │  ├─ generator/
│  │  │  ├─ index.js         Generator ファクトリ（抽象）
│  │  │  └─ geminiGenerator.js
│  │  └─ pipeline.js         一連のオーケストレーション
│  ├─ server/
│  │  ├─ app.js              Express・ルート・静的配信
│  │  ├─ auth.js             認証ミドルウェア
│  │  └─ rateLimit.js        レート制限
│  └─ index.js               サーバー起動エントリ
├─ scripts/
│  ├─ buildIndex.js          索引構築（Python rebuild_index 相当）
│  ├─ search.js              検索のみ確認（try_hybrid 相当）
│  └─ evaluate.js            評価（Phase 2）
├─ public/                   Alpine.js UI
│  ├─ index.html
│  ├─ app.js
│  └─ style.css
├─ .env.example
├─ package.json
└─ README.md
```

---

## 4. モジュール設計（責務と公開インターフェース）

関数シグネチャは実装の契約として示す。すべて ES Modules（import/export）想定。

### 4.1 config.js
```
export const config = {
  docsDir, indexPath, dictDir, modelPath,
  embeddingModel, topK, candidateK,
  geminiApiKey, geminiModel, maxOutputTokens, temperature,
  authToken, rateLimit: { windowMs, max },
};
```

### 4.2 embedder.js（Python embedder.py 相当）
```
createEmbedder({ modelPath }) -> {
  embed(texts: string[]): Promise<number[][]>,
  embedSingle(text: string): Promise<number[]>,
}
```
- transformers.js の pipeline('feature-extraction', model) を用い、mean pooling + 正規化で文ベクトルを得る。
- モデルはローカル配置（models/）。初回ロードを一度だけ行い使い回す。

### 4.3 vectorStore.js（Python retriever.py 相当・Chroma置換）
```
loadVectorStore(indexPath) -> {
  chunks,                          // [{id, source, folder, page, chunkIdx, text, embedding}]
  search(queryVec, k): Array<{ id, text, metadata, distance }>,
}
```
- 起動時に index.json を読み込み、各 embedding のノルムを前計算。
- search はコサイン類似度を全件計算し上位 k を返す（distance = 1 - cos）。

### 4.4 tokenizer.js（Python tokenizer.py 相当・用語保護層を内蔵）
```
createTokenizer({ protectTermsPath }) -> {
  tokenize(text: string): Promise<string[]>,   // 内容語の原形・小文字
}
```
- kuromoji.js で分かち書きし、助詞・記号を除く。
- 用語保護層: protect_terms.txt の語が誤分割されないよう、トークナイズ前にプレースホルダ化→後で復元、もしくは語境界を強制し1語として返す。

### 4.5 bm25.js（Python bm25_retriever.py 相当）
```
buildBm25(docsTokens: string[][]) -> {
  search(queryTokens: string[], k): Array<{ index, score }>,
}
```
- BM25（Okapi）。索引構築は起動時に vectorStore の chunks から分かち書きして行う。

### 4.6 glossary.js（Python glossary.py 相当）
```
loadGlossary(glossaryMdPath) -> {
  terms: Map<alias, definition>,
  expand(query): { query: string, matched: string[] },  // 一致語の定義を連結
}
```

### 4.7 hybrid.js（Python hybrid_retriever.py 相当）
```
createHybrid({ embedder, vectorStore, bm25, tokenizer, glossary, kakariTerms, rrfK=60, boostWeight=0.03 }) -> {
  search(query, { topK, candidateK }): Promise<Array<{ id, text, metadata, rrfScore, boosted }>>,
}
```
処理順:
1. 用語注入: glossary.expand(query) で拡張クエリを得る。
2. 拡張クエリで embedder.embedSingle と tokenizer→bm25.search。
3. ベクトル候補・BM25候補を RRF（1/(k+rank)）で統合。
4. 係名ブースト: 元の質問に含まれる係名が出典ファイル名に一致したら boostWeight を加点。
5. スコア降順で topK。

### 4.8 contextFormatter.js（Python prompt_templates.py 相当）
```
formatContext(results): string     // ① 同一文書まとめ＋文書順 ② 隣接結合 ③ ヘッダー整理
formatSources(results): Array<{ source, folder, page }>
SYSTEM_PROMPT: string
```

### 4.9 generator/（Python generator.py 相当・抽象化）
```
// generator/index.js
createGenerator(config) -> generator   // config.generator により実装を選択（既定: gemini）

// generator/geminiGenerator.js
createGeminiGenerator({ apiKey, model, maxOutputTokens, temperature }) -> {
  generate({ system, context, query }): Promise<{ text, usage }>,
}
```
- Gemini 実装は @google/genai を用い、thinkingConfig で思考を無効化（思考トークンによる途中切れ対策）。
- 将来 localLlmGenerator.js / extractiveGenerator.js を同じインターフェースで追加可能。

### 4.10 pipeline.js（Python pipeline.py 相当）
```
createPipeline({ hybrid, generator }) -> {
  ask(query, { topK }): Promise<{ query, answer, sources, searchResults, metrics }>,
}
```
- metrics: { searchTime, generationTime, totalTime, tokens, numResults }。

### 4.11 server/
```
app.js     : Express。GET / で public 配信、POST /api/ask・/api/search。auth・rateLimit を適用。
auth.js    : Authorization トークン（config.authToken）を検証。
rateLimit.js: IP/セッション単位の呼び出し上限。
index.js   : 起動時に embedder/vectorStore/bm25/glossary/hybrid/generator/pipeline を構築し app を起動。
```

---

## 5. データ形式

### 5.1 index.json
```
{
  "model": "paraphrase-multilingual-MiniLM-L12-v2",
  "dim": 384,
  "builtAt": "2026-06-16T...",
  "chunks": [
    {
      "id": "総リーのお仕事_4_1_ab12cd34",
      "source": "総リーのお仕事.pdf",
      "folder": "ねっしー遺産/3.マニュアルあれこれ/2.通常イベント",
      "page": 4,
      "chunkIdx": 1,
      "text": "【総リーのお仕事】\n...",
      "embedding": [0.01, -0.02, ...]   // dim 個
    }
  ]
}
```
- text には Python版同様、先頭に係名ヘッダー【...】を含める（埋め込み・表示の整合のため）。
- 規模見積り: 約335チャンク×384次元。JSONで数MB程度。将来は Float32 のバイナリ化で軽量化可能（最適化は後回し）。

### 5.2 辞書類（data/dict/）
- kakari_terms.txt: 1行1語の係名（# はコメント）。Python版 kakari_terms.csv を流用。
- protect_terms.txt: 用語保護層で1語扱いする固有語（係名＋用語辞典見出し語をまとめる）。
- glossary.md: 用語辞典（## 見出し＝用語）。用語注入の定義源。

### 5.3 設定（.env）
```
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=Xenova/paraphrase-multilingual-MiniLM-L12-v2
TOP_K=10
AUTH_TOKEN=...
RATE_LIMIT_MAX=30
```

---

## 6. データフロー

### 6.1 索引構築（scripts/buildIndex.js）
```
data/docs の PDF/MD を走査
  └ 同名 .md があれば PDF をスキップ（.md 優先）
  └ ページ抽出（PDF=pdf-parse, MD=全体を1ページ）
  └ normalizeText（連続空白圧縮・空行除去）
  └ チャンク分割（MD=見出し単位 / PDF=固定長500・overlap50）
  └ 薄いチャンク除外（非空白 < 30字）
  └ 係名ヘッダー付与（【ファイル名】）
  └ embedder.embed でベクトル化
  └ index.json に保存
```

### 6.2 質問応答（POST /api/ask）
```
query
  └ auth・rateLimit
  └ pipeline.ask(query)
       └ hybrid.search(query)
            └ glossary.expand → 拡張クエリ
            └ embedSingle(拡張) + tokenize(拡張)→bm25
            └ vectorStore.search + bm25.search
            └ RRF 統合
            └ 係名ブースト（元query）
            └ topK
       └ contextFormatter.formatContext
       └ generator.generate（Gemini, 思考無効化）
       └ formatSources
  └ JSON 応答 { answer, sources, metrics }
```

---

## 7. API仕様

### POST /api/ask
- 認証: Authorization ヘッダ（トークン）。レート制限あり。
- リクエスト: `{ "query": "総リーの仕事は？" }`
- レスポンス:
```
{
  "answer": "総リーの仕事は...",
  "sources": [{ "source": "総リーのお仕事.pdf", "folder": "...", "page": 4 }],
  "metrics": { "searchTime": 0.02, "generationTime": 3.5, "totalTime": 3.6,
               "tokens": { "input": 2731, "output": 339, "total": 3070 } }
}
```

### POST /api/search（検索のみ・LLM不要）
- リクエスト: `{ "query": "..." }`
- レスポンス: `{ "results": [{ source, folder, page, rrfScore, boosted, text }] }`

### GET /
- public/index.html（Alpine.js UI）を返す。静的アセットは public/ から配信。

---

## 8. 検索アルゴリズム（移植仕様の確認）

Python版で確定したロジックをそのまま移す。

- RRF: 各リトリーバの順位 rank（1始まり）について `1/(rrfK + rank)` を加算（rrfK=60）。
- 係名ブースト: 元の質問に含まれる係名 t について、チャンクの source が t を含むなら rrfScore に boostWeight(0.03) を加点。判定は拡張クエリではなく元質問で行う（定義文中の他係名で誤ブーストしないため）。
- 用語注入: glossary の各 alias が質問に含まれれば、その定義を連結した拡張クエリを検索に用いる。定義の重複は除く。
- 文脈整形: ① source でグループ化（関連度順を保持）② グループ内を (page, chunkIdx) 昇順、隣接（同ページ・連番）を重複除去して結合 ③ 本文先頭の【係名】を除き出典はグループ見出しで一度だけ。
- 用語保護層: protect_terms の語をトークナイズで1語に保つ（Janome ユーザー辞書の代替）。

---

## 9. セキュリティ設計

| 対策 | 実装 |
|------|------|
| 認証 | Authorization トークン（config.authToken）を全 /api に必須化。auth.js |
| 通信暗号化 | TLS は Tunnel 側（Tailscale Funnel / Cloudflare Tunnel）で終端 |
| APIキー保護 | Gemini キーはサーバー内のみ。フロントへ出さない |
| レート制限 | IP/トークン単位で windowMs あたり max 回。rateLimit.js |
| 入力検証 | query の最大長・型チェック。空・過大入力を弾く |

---

## 10. 設定・起動・運用

- 索引構築: `node scripts/buildIndex.js`（文書追加・更新時に実行）
- 検索確認: `node scripts/search.js "総リーの仕事は"`
- サーバー起動: `node src/index.js`（pm2 等で常駐化）
- 公開: Tunnel で外部公開（ポート開放なし）
- バックアップ: data/（docs・index.json・dict）を定期バックアップ

---

## 11. 移植上の注意点

- 埋め込みの数値差: transformers.js(ONNX) と Python(PyTorch) で出力に微差が出るため、索引は必ずJS側の embedder で再構築する。質問と文書の埋め込みは同一実装で揃える。
- 量子化: モデルは量子化ONNXを基本とし、速度・メモリと精度のバランスを動作確認で調整（残課題）。
- kuromoji 辞書: 内蔵IPADICでは固有語が誤分割されるため、用語保護層を必ず通す。
- index.json サイズ: 規模拡大時はバイナリ埋め込みへ移行を検討。
- 評価: Python版 evaluate 相当を Phase 2 で移植し、移植前後で Hit@k・MRR を比較して品質劣化がないか確認する。

---

## 12. 実装順序（Phase 1）

1. config / embedder / vectorStore
2. scripts/buildIndex（索引構築して index.json を作る）
3. tokenizer（用語保護層）/ bm25 / glossary / hybrid
4. contextFormatter / generator(Gemini) / pipeline
5. scripts/search で検索・回答をCLI確認
6. server（API・認証・レート制限）＋ public（Alpine.js UI）
7. ラズパイ動作確認・チューニング
