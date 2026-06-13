# RAGシステム 技術設計書

**作成日**: 2026-06-12  
**バージョン**: 1.0

---

## 1. システムアーキテクチャ

### 1.1 全体図

```
┌──────────────────────────────────────────────────────────┐
│  ユーザー（Tailscale/VPN経由）                             │
└──────────────┬───────────────────────────────────────────┘
               │ HTTPS
┌──────────────▼───────────────────────────────────────────┐
│  ラズパイ（部室内、常時稼働）                               │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─ フロントエンド層 ────────────────────┐              │
│  │ React (TypeScript + Vite)           │              │
│  │ - チャット UI                       │              │
│  │ - 認証・セッション管理              │              │
│  └────────────────┬────────────────────┘              │
│                   │ HTTP/JSON                         │
│  ┌────────────────▼────────────────────┐              │
│  │ バックエンド層（FastAPI）           │              │
│  │ ┌──────────────────────────────┐   │              │
│  │ │ Web Server (uvicorn)         │   │              │
│  │ ├──────────────────────────────┤   │              │
│  │ │ ルータ・認証・バリデーション  │   │              │
│  │ ├──────────────────────────────┤   │              │
│  │ │ RAG パイプライン             │   │              │
│  │ │ - Chroma 検索                │   │              │
│  │ │ - Gemini API 呼び出し        │   │              │
│  │ │ - 出典追跡                   │   │              │
│  │ └──────────────────────────────┘   │              │
│  └────────────────┬────────────────────┘              │
│                   │                                   │
│  ┌────────────────▼────────────────────┐              │
│  │ ローカルストレージ層                 │              │
│  │ ┌──────────────────────────────┐   │              │
│  │ │ Chroma DB                    │   │              │
│  │ │ (ベクトルDB + メタデータ)    │   │              │
│  │ └──────────────────────────────┘   │              │
│  │ ┌──────────────────────────────┐   │              │
│  │ │ SQLite                       │   │              │
│  │ │ (セッション・PDF メタデータ) │   │              │
│  │ └──────────────────────────────┘   │              │
│  │ ┌──────────────────────────────┐   │              │
│  │ │ PDF ストレージ               │   │              │
│  │ │ (/data/pdfs/)               │   │              │
│  │ └──────────────────────────────┘   │              │
│  └────────────────────────────────────┘              │
│                                                       │
│  ┌──────────────────────────────────┐                │
│  │ バックアップ・ログ              │                │
│  │ - Chroma スナップショット        │                │
│  │ - PDF ファイル定期バックアップ   │                │
│  │ - アクセスログ・エラーログ       │                │
│  └──────────────────────────────────┘                │
│                                                      │
└──────────────────────────────────────────────────────┘
                      ↓ HTTPS
         ┌────────────────────────┐
         │ 外部 API（インターネット）│
         ├────────────────────────┤
         │ - Gemini API           │
         │ - HuggingFace          │
         │  （モデルメタ取得）    │
         └────────────────────────┘
```

---

## 2. データフロー

### 2.1 質問応答フロー

```
1. ユーザー入力
   ┌─────────────────┐
   │ "犬について教えて" │
   └────────┬────────┘
            │
2. FastAPI でリクエスト受信
   ├─ 認証・セッション確認
   ├─ 入力値バリデーション
   └────────┬────────┘
            │
3. Embedding（質問をベクトル化）
   ├─ sentence-transformers (all-MiniLM-L6-v2)
   ├─ ローカルで実行（外部API呼び出しなし）
   └────────┬────────┘
            │ [0.12, -0.45, 0.89, ...]
4. Chroma で検索
   ├─ 質問ベクトルに最も近い PDF テキスト片を抽出
   ├─ Top-K (K=3〜5) で関連箇所を取得
   └────────┬────────┘
            │ [
            │   {text: "犬は...", pdf: "manual.pdf", page: 5},
            │   {text: "犬の特徴...", pdf: "manual.pdf", page: 6},
            │   ...
            │ ]
5. プロンプト組み立て
   ├─ システムプロンプト（ハルシネーション抑止）
   ├─ 検索結果を「参考情報」として追加
   ├─ ユーザー質問を追加
   └────────┬────────┘
            │
6. Gemini API で生成
   ├─ POST /api/chat/completions (Gemini)
   ├─ max_tokens=500, temperature=0.3
   └────────┬────────┘
            │ {text: "犬は..."}
7. 出典追跡・マッピング
   ├─ 生成テキストと検索結果を照合
   ├─ 参照 PDF・ページ番号を確定
   └────────┬────────┘
            │
8. レスポンス組み立て
   ├─ 生成テキスト + 出典情報
   └────────┬────────┘
            │ {
            │   answer: "犬は...",
            │   sources: [
            │     {pdf: "manual.pdf", page: 5},
            │     {pdf: "manual.pdf", page: 6}
            │   ]
            │ }
9. ユーザーに返信
   └─ React UI で表示
```

### 2.2 PDF インデックス構築フロー

```
管理者: スクリプト実行
$ python scripts/rebuild_index.py --pdfs /data/pdfs/

  ↓

1. PDF 読み込み
   ├─ pypdf で全ページ抽出
   ├─ テキスト正規化（改行・空白）
   └─ ページ単位でメタデータ付与
      {text: "...", source: "manual.pdf", page: 5}

  ↓

2. チャンク分割
   ├─ 1 チャンク = 500 文字程度
   ├─ オーバーラップ = 50 文字（文脈保持）
   └─ メタデータ保持（どの PDF のどのページか）

  ↓

3. Embedding（テキスト→ベクトル化）
   ├─ sentence-transformers で各チャンクをベクトル化
   ├─ ローカルで実行（外部API呼び出しなし）
   └─ 出力: [0.12, -0.45, 0.89, ...] (384次元)

  ↓

4. Chroma に登録
   ├─ ベクトル + メタデータを格納
   ├─ インデックス更新
   └─ 既存索引の置き換え

  ↓

5. SQLite にメタデータ記録
   ├─ PDF ファイル名・更新日時
   ├─ チャンク数・最後のインデックス更新日時
   └─ バージョン情報

  ↓

管理者: 完了通知 ✓
```

---

## 3. フォルダ構成

```
nec_chatbot/
├── backend/                          # FastAPI アプリケーション
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI アプリケーション（Web UI は後実装）
│   │   ├── config.py                 # 環境変数・設定管理
│   │   ├── dependencies.py           # DI（Chroma, モデル初期化）
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── chat.py               # チャットエンドポイント（後実装）
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── embedder.py           # Embedding（sentence-transformers）
│   │   │   ├── retriever.py          # Chroma 検索
│   │   │   ├── generator.py          # Gemini API 呼び出し
│   │   │   ├── pipeline.py           # RAG パイプライン全体 ★重点実装
│   │   │   └── prompt_templates.py   # プロンプト定義
│   │   ├── models.py                 # Pydantic モデル
│   │   └── utils.py                  # ユーティリティ関数
│   ├── requirements.txt               # Python 依存関係
│   └── pyproject.toml                 # Poetry 設定
│
├── scripts/                          # 実験・運用用スクリプト
│   ├── rebuild_index.py               # PDF インデックス再構築スクリプト
│   ├── test_rag.py                   # RAG パイプラインテスト（ターミナル実行）
│   └── cli.py                        # 簡易 CLI（ターミナルで質問実験）
│
├── notebooks/                        # Jupyter Notebook（実験用）
│   ├── 01_embedding_test.ipynb       # Embedding モデル検証
│   ├── 02_chroma_test.ipynb          # Chroma 検索テスト
│   ├── 03_rag_pipeline_test.ipynb    # RAG パイプライン全体テスト
│   └── 04_gemini_integration.ipynb   # Gemini API 統合テスト
│
├── data/                             # ローカルストレージ
│   ├── pdfs/                         # PDF ファイル（サンプル）
│   │   ├── sample1.pdf
│   │   └── sample2.pdf
│   ├── chroma_db/                    # Chroma ベクトルDB
│   │   └── （実行時に生成）
│   └── backups/                      # バックアップ
│
├── docs/                             # ドキュメント
│   ├── requirements.md               # 要件定義
│   ├── technical_design.md          # 技術設計（このファイル）
│   ├── SETUP.md                      # セットアップ手順
│   └── OPERATION.md                  # 運用マニュアル（後で作成）
│
├── requirements.txt                  # Python 依存関係（統一）
├── .env.example                      # 環境変数テンプレート（統一）
├── .gitignore
└── README.md                         # プロジェクト概要

```

**フロントエンド実装について**  
Webインターフェース（React）は **Phase 2** として後回しに。現段階では：
- ターミナル CLI で実験
- Jupyter Notebook で検証
- RAG パイプラインの精度・精緻化に集中

---

## 4. モジュール設計

### 4.1 バックエンド

#### embedder.py
```python
class Embedder:
    """sentence-transformers を使用したテキスト埋め込み"""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # モデルのダウンロード・初期化（初回のみ）
        # 以降はキャッシュから読み込み
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """テキストをベクトルに変換"""
        # 質問またはチャンクをベクトル化
```

#### retriever.py
```python
class Retriever:
    """Chroma を使用したベクトル検索"""
    def __init__(self, chroma_path: str):
        # Chroma クライアント初期化
    
    def search(self, query_vector: List[float], top_k: int = 5):
        """質問ベクトルで関連テキストチャンクを検索
        
        Returns:
            [{text: "...", source: "manual.pdf", page: 5}, ...]
        """
```

#### generator.py
```python
class Generator:
    """Gemini API で回答生成"""
    def __init__(self, api_key: str):
        # Gemini API クライアント初期化
    
    def generate(self, prompt: str, max_tokens: int = 500):
        """プロンプトから回答を生成"""
        # Gemini API 呼び出し
```

#### pipeline.py
```python
class RAGPipeline:
    """RAG の全体フロー"""
    def __init__(self, embedder, retriever, generator):
        pass
    
    def query(self, user_question: str, top_k: int = 5):
        """ユーザー質問 → 検索 → 生成 → 出典付き回答"""
        # 1. 質問をベクトル化
        # 2. Chroma で関連テキストを検索
        # 3. プロンプト組み立て
        # 4. Gemini で生成
        # 5. 出典情報を付加
        # return {answer, sources}
```

### 4.2 API エンドポイント

#### `/api/chat`
```
POST /api/chat
Content-Type: application/json

{
  "message": "犬について教えて",
  "session_id": "..."
}

Response (200):
{
  "id": "msg-123",
  "answer": "犬は...",
  "sources": [
    {"pdf": "manual.pdf", "page": 5},
    {"pdf": "rule.pdf", "page": 3}
  ],
  "timestamp": "2026-06-12T10:30:00Z"
}
```

#### `/api/admin/rebuild-index` (管理者用)
```
POST /api/admin/rebuild-index
Authorization: Bearer <admin_token>

Response (200):
{
  "status": "success",
  "indexed_pdfs": 5,
  "total_chunks": 1250,
  "timestamp": "2026-06-12T10:30:00Z"
}
```

---

## 5. 実装フェーズ

### Phase 1: RAG パイプライン構築（優先）
**目標**: ターミナル/Jupyter で RAG を実験可能な状態に

- [ ] Python 環境セットアップ（requirements.txt）
- [ ] sentence-transformers + Chroma セットアップ
- [ ] PDF 読み込み・チャンク分割 (pypdf)
- [ ] Embedding + インデックス構築スクリプト実装
- [ ] Gemini API 統合
- [ ] RAG パイプライン実装
- [ ] ターミナル CLI (test_rag.py, cli.py) で検証
- [ ] Jupyter Notebook での段階的テスト

### Phase 2: Web UI（後回し）
**目標**: REST API + React フロントエンド実装

- [ ] FastAPI で RAG パイプライン API化
- [ ] 認証・セッション管理の実装
- [ ] React + TypeScript でチャット UI
- [ ] API 連携・出典表示機能

### Phase 3: ラズパイデプロイ・運用（最後）
**目標**: ラズパイでの常時稼働・バックアップ体制

- [ ] ラズパイへのセットアップスクリプト
- [ ] バックアップ・ロギング実装
- [ ] Tailscale / VPN 接続確認
- [ ] 運用マニュアル作成

---

## 6. セキュリティ考慮事項

| 項目 | 対策 |
|------|------|
| **プライバシー** | PDF 全文はラズパイ内、API には質問+検索結果のみ送信 |
| **認証** | 既存認証システム統合、セッション管理 |
| **HTTPS** | ラズパイで SSL 証明書設定（自己署名 OK） |
| **API キー管理** | .env ファイル（git ignore）で管理 |
| **ログ** | 質問内容は記録しない（プライバシー） |

---

## 7. パフォーマンス想定

| 処理 | 想定時間 | 備考 |
|------|---------|------|
| Embedding（質問） | 100-200ms | ローカル（ネットワーク遅延なし） |
| Chroma 検索 | 50-100ms | Top-5 抽出 |
| Gemini API 呼び出し | 1-3秒 | ネットワーク遅延含む |
| トータル | 1.5-3.5秒 | ネットワーク品質に左右される |

---

## 参考・関連ファイル

- 要件定義: [requirements.md](requirements.md)
- デプロイメント: [DEPLOYMENT.md](DEPLOYMENT.md) (未作成)
- 運用マニュアル: [OPERATION.md](OPERATION.md) (未作成)
