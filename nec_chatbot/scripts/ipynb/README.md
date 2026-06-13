# RAG パイプライン 検証環境

ターミナルの CLI の不便さを解消するため、Jupyter Notebook ベースの実験環境です。

## 特徴

- **インタラクティブな質問入力** - セル内で何度でも修正・再実行可能
- **処理ステップの詳細表示** - エンベディング、検索、生成 各段階の処理時間を記録
- **トークン使用量の可視化** - 入力・出力・合計トークン数を表示
- **検索結果の詳細確認** - ドキュメントのスコアと内容を確認
- **複数クエリの一括実行** - テスト用複数質問の比較検証

## 使い方

### 1. 環境準備

```bash
# Jupyter Notebook のインストール（必要な場合）
pip install jupyter

# 仮想環境の有効化
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# リポジトリルートから実行
cd nec_chatbot/scripts/ipynb
```

### 2. Notebook 起動

```bash
jupyter notebook experiment.ipynb
```

ブラウザで `http://localhost:8888` が開きます。

### 3. セルの実行順序

1. **セル 1: インポート** - ライブラリとモジュールの読み込み
2. **セル 2: 初期化** - RAG パイプラインの初期化
3. **セル 3: 補助関数定義** - フォーマッティング関数の定義
4. **セル 4: 単一クエリ実行** ← ここで質問を入力（重要）
5. **セル 5: 複数クエリ実行** - 複数質問の一括テスト
6. **セル 6: 検索のみモード** - 検索結果だけを確認

### 4. 質問を入力する

セル 4 の最初の行を修正：

```python
query = "あなたの質問をここに入力"  # ← これを変更
```

実行すると、以下の情報が表示されます：

- 📋 **回答** - Gemini API が生成した回答
- 📌 **出典** - 参考にしたドキュメント
- ⏱️ **処理時間**
  - エンベディング処理時間
  - 検索処理時間
  - 生成処理時間
  - **合計処理時間**
- 🔤 **トークン使用量**
  - 入力トークン数
  - 出力トークン数
  - **合計トークン数**
- 📚 **検索結果（詳細）**
  - ドキュメント名・ページ番号
  - スコア（関連度）
  - 該当テキスト抜粋

## セルの詳細説明

### セル 4: 単一クエリ実行

```python
query = "質問を入力"
result = pipeline.query_with_info(query, top_k=config.CHROMA_TOP_K)
```

**出力内容:**

```
🔍 質問: ...

====
📋 回答
====
[生成された回答が表示]

====
📌 出典
====
- docs/guide.md (p.5)
- docs/reference.md (p.12)
...

====
⏱️ 処理時間（秒）
====
  エンベディング: 0.234s
  検索:          0.156s
  生成:          2.451s
  ---
  合計:          2.841s

====
🔤 トークン使用量
====
  入力:  512
  出力:  248
  ---
  合計:  760

====
📚 検索結果（詳細）
====
[スコア付きで検索結果を表示]
```

### セル 5: 複数クエリの実行

複数の質問を同時にテストする場合：

```python
test_queries = [
    "質問1",
    "質問2",
    "質問3",
]
```

結果をサマリーテーブルで比較表示します。

### セル 6: 検索のみモード

回答生成をスキップして、検索結果のみを高速に確認：

```python
search_query = "検索クエリ"
results = retriever.search(query_embedding, top_k=config.CHROMA_TOP_K)
```

## 便利なテクニック

### セルを何度も実行

```python
# セル 4 の query の値を変更して、Shift+Enter で再実行
query = "別の質問"
```

実行ボタンはセル右上、またはキーボード `Shift+Enter`。

### 実行時間の短縮

```python
# top_k を小さくして、検索結果数を減らす
result = pipeline.query_with_info(query, top_k=3)
```

### 環境変数の確認

セル 2 で設定情報が表示されます。トラブルシューティング時に確認してください。

## トラブルシューティング

### Chroma DB が見つからない

```
❌ Chroma DB が見つかりません
```

対応方法：

```bash
# リポジトリルートから
python scripts/rebuild_index.py
```

### モジュール import エラー

```
ModuleNotFoundError: No module named 'app.rag'
```

対応方法：

1. セル 1 の `backend_path` が正しいか確認
2. 仮想環境が有効か確認：`pip list | grep google-generativeai`

### Gemini API エラー

```
⚠️ GEMINI_API_KEY が設定されていません
```

対応方法：

```bash
# .env ファイルで API キーを設定
# または
export GEMINI_API_KEY="your-key-here"
```

## パラメータの説明

| パラメータ | 説明 | デフォルト |
|-----------|------|----------|
| `query` | ユーザーの質問 | - |
| `top_k` | 検索結果の上位 K 件数 | 5 |
| `max_tokens` | 生成の最大トークン数 | 500 |
| `temperature` | 温度（0-1、低いほど確定的） | 0.3 |

## メトリクスの見方

### 処理時間

- **エンベディング** - 質問をベクトル化する時間
- **検索** - ベクトル検索で関連ドキュメントを探す時間
- **生成** - Gemini API が回答を生成する時間（通常最長）

### トークン使用量

- **入力** - プロンプト（質問+参考情報）のトークン数
- **出力** - 生成された回答のトークン数
- **合計** - API の請求対象となる合計数

Gemini の料金モデルについては、公式ドキュメントを参照：
https://ai.google.dev/pricing

## 参考資料

- [Gemini API ドキュメント](https://ai.google.dev/docs)
- [Chroma DB](https://docs.trychroma.com/)
- [Jupyter Notebook](https://jupyter.org/)
