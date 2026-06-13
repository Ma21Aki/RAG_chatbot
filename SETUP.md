# セットアップ手順書（開発仲間向け）

このプロジェクト（RAGチャットボット）をローカルで動かすための手順です。
上から順に進めれば、質問応答の実験ができる状態になります。

所要時間の目安: 15〜30分（依存インストールとモデルの初回ダウンロード含む）

---

## 0. 事前に用意するもの

- Python 3.12 系（このプロジェクトは 3.12.8 で動作確認）
  - 確認: `python --version`
- Git
- プロジェクト管理者から別途受け取るもの（GitHubには入っていません）
  - データ一式の zip（PDF と 検索インデックス chroma_db）
  - 自分用の Gemini API キー（後述。各自で取得する方式でもOK）

補足: コードは GitHub の private リポジトリにあります。管理者にあなたの
GitHub アカウントを伝え、コラボレーター招待を承認してから進めてください。

---

## 1. リポジトリを取得（clone）

```powershell
git clone https://github.com/＜管理者のアカウント＞/＜リポジトリ名＞.git
cd ＜リポジトリ名＞
```

clone 後のフォルダ構成（抜粋）:

```
＜リポジトリ名＞/        ← ここがプロジェクトルート。.env はここに置く
├─ .env.example
├─ SETUP.md            ← この手順書
└─ nec_chatbot/
   ├─ backend/         ← venv はこの中に作る
   │  └─ requirements.txt
   ├─ data/            ← PDF と chroma_db はここに配置
   └─ scripts/
```

---

## 2. データを配置する（管理者から受け取った zip）

受け取った zip を解凍し、中身を `nec_chatbot/data/` の下に置きます。
最終的に以下の形になればOKです。

```
nec_chatbot/data/
├─ pdfs/        ← PDF 群（例: ねっしー遺産/ など）
└─ chroma_db/   ← 検索インデックス（中に UUID のフォルダ群と chroma.sqlite3）
```

chroma_db を配置しておけば、再インデックス（rebuild_index.py）は不要で
そのまま検索できます。

---

## 3. 仮想環境（venv）を作る

backend フォルダ内に venv を作ります。

```powershell
cd nec_chatbot\backend
python -m venv venv
```

有効化（PowerShell）:

```powershell
.\venv\Scripts\Activate.ps1
```

うまくいくとプロンプトの先頭に `(venv)` が付きます。

PowerShell でスクリプト実行がブロックされる場合は、一度だけ次を実行してから
再度有効化してください。

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

（コマンドプロンプト cmd を使う場合は `venv\Scripts\activate.bat`）

---

## 4. 依存ライブラリをインストール

venv を有効化した状態で、backend フォルダにて:

```powershell
pip install -r requirements.txt
```

主に入るもの: chromadb（ベクトルDB）、sentence-transformers（埋め込み）、
pypdf、google-genai（Gemini）、jupyter など。

補足: 埋め込みモデル（paraphrase-multilingual-MiniLM-L12-v2、約470MB）は、
最初に実行したときに自動で HuggingFace からダウンロードされます。初回のみ
ネット接続と数分の待ち時間が必要です。

---

## 5. APIキーを設定する（.env の作成）

回答生成に Gemini API を使うため、API キーが必要です。

### 5-1. キーを取得（各自で用意する場合）

1. https://aistudio.google.com/apikey にアクセス
2. Google アカウントでログインし「APIキーを作成」
3. 表示されたキー（`AIza...` 形式）をコピー

注意: 無料枠を超える場合や生成が 429（クレジット枯渇）で止まる場合は、
請求（billing）の設定が必要です。

### 5-2. .env を作る

プロジェクトルート（リポジトリ直下、SETUP.md と同じ階層）で、`.env.example`
をコピーして `.env` を作ります。

```powershell
# プロジェクトルートに移動してから
copy .env.example .env
```

`.env` をエディタで開き、キーを書き換えます。

```
GEMINI_API_KEY=ここに自分のAPIキーを貼る
```

他の項目（EMBEDDING_MODEL_NAME, CHROMA_TOP_K など）は基本そのままでOKです。

重要: `.env` は秘密情報なので Git にコミットしない（.gitignore で除外済み）。
キーをチャットやコミットに貼らないこと。

---

## 6. 動作確認

### 方法A: Jupyter ノートブックで試す（おすすめ）

1. VSCode で `nec_chatbot/scripts/ipynb/experiment.ipynb` を開く
2. 右上のカーネル選択で、手順3で作った venv
   （`nec_chatbot/backend/venv`）の Python を選ぶ
3. 上のセルから順に実行（cell-1 → cell-3 で初期化）
4. 質問セルで次のように呼ぶと回答が出る

```python
ask_to_chatbot("鉈を使うときの注意点は？")
```

Gemini を使わず検索だけ確認したいときは:

```python
serche_query("肝試し")
```

cell-3 の出力で次が出ていれば初期化成功です。

```
✓ Chroma DB: ...\data\chroma_db
✓ Embedding モデル: paraphrase-multilingual-MiniLM-L12-v2
```

### 方法B: ターミナル CLI で試す

venv 有効化状態で、nec_chatbot フォルダから:

```powershell
.\backend\venv\Scripts\python.exe scripts\cli.py
```

質問を入力すると回答が返ります（`exit` で終了）。

---

## 7. よくあるつまずき

- `NameError: name 'pipeline' is not defined`（ノートブック）
  → 初期化セル（cell-3）を実行していない。上のセルから順に実行する。

- `❌ Chroma DB が見つかりません`
  → 手順2のデータ配置ができていない。`nec_chatbot/data/chroma_db/` を確認。

- 回答が `429 RESOURCE_EXHAUSTED` / `prepayment credits` で止まる
  → Gemini の無料枠超過または請求未設定。AI Studio で billing を確認。
    検索だけ試したいなら `serche_query()` は Gemini 不要なので動く。

- import エラー（app.rag が見つからない 等）
  → venv を有効化しているか、`pip install -r requirements.txt` 済みかを確認。

- PowerShell で `Activate.ps1` が実行できない
  → 手順3の `Set-ExecutionPolicy ... RemoteSigned` を実行。

---

## まとめ（最短経路）

1. clone → 2. データzipを data/ に展開 → 3. venv作成・有効化 →
4. `pip install -r requirements.txt` → 5. `.env` にAPIキー →
6. ノートブックで `ask_to_chatbot("質問")`
