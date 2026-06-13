# Gemini API 使い方ガイド

## パッケージについて

`google-generativeai` は2025年8月末に廃止済み。
このプロジェクトでは新しい `google-genai` を使用する。

```
# requirements.txt
google-genai>=2.0.0
```

---

## 基本的な書き方

### インポートとクライアント作成

```python
from google import genai
from google.genai import types

# API キーを直接渡す場合
client = genai.Client(api_key="YOUR_API_KEY")

# 環境変数 GEMINI_API_KEY から自動取得する場合
client = genai.Client()
```

---

### テキスト生成

```python
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="ここに質問を入力",
    config=types.GenerateContentConfig(
        max_output_tokens=500,
        temperature=0.3,  # 0.0〜1.0。低いほど決定的な回答
    )
)

print(response.text)
```

---

### トークン使用量の取得

```python
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="ここに質問を入力",
)

usage = response.usage_metadata
print(usage.prompt_token_count)      # 入力トークン数
print(usage.candidates_token_count)  # 出力トークン数
print(usage.total_token_count)       # 合計トークン数
```

---

### ストリーミング生成

```python
for chunk in client.models.generate_content_stream(
    model="gemini-2.0-flash",
    contents="ここに質問を入力",
    config=types.GenerateContentConfig(
        max_output_tokens=500,
        temperature=0.3,
    )
):
    if chunk.text:
        print(chunk.text, end="", flush=True)
```

---

## 旧APIとの対応表

| 項目 | 旧: `google-generativeai` | 新: `google-genai` |
|------|--------------------------|-------------------|
| インポート | `import google.generativeai as genai` | `from google import genai` |
| 初期化 | `genai.configure(api_key=...)` | `genai.Client(api_key=...)` |
| モデル作成 | `genai.GenerativeModel("gemini-pro")` | 不要（呼び出し時に指定） |
| 生成 | `model.generate_content(prompt)` | `client.models.generate_content(model=..., contents=...)` |
| 設定クラス | `genai.types.GenerationConfig` | `types.GenerateContentConfig` |
| ストリーミング | `model.generate_content(..., stream=True)` | `client.models.generate_content_stream(...)` |
| トークン合計 | `prompt_token_count + candidates_token_count` | `usage.total_token_count` |

---

## このプロジェクトでの実装箇所

- [backend/app/rag/generator.py](../backend/app/rag/generator.py) — `Generator` クラス
- [backend/requirements.txt](../backend/requirements.txt) — 依存パッケージ

## 推奨モデル名

| モデル | 用途 |
|--------|------|
| `gemini-2.5-flash` | 高精度が必要な場合 |

注意: 利用可能なモデルはAPIキーの契約プランによって異なる。
`client.models.list()` で確認できる。
