# rag/query.py
import google.generativeai as genai
import numpy as np

# 🔐 ここにあなたのAPIキーを直接書く！
genai.configure(api_key="AIzaSyC6MGicISfQ6CqIOTJfVs1lNadsm8N3VLQ")

# 軽量で無料枠向きのモデル
model = genai.GenerativeModel("gemini-1.5-flash")

def retrieve_and_answer(user_query, index, chunks, embed_model, top_k=3):
    query_vec = embed_model.encode([user_query])
    D, I = index.search(np.array(query_vec), top_k)
    context = "\n".join([chunks[i] for i in I[0]])

    prompt = f"""以下の文書に基づいて質問に答えてください。

文書:
{context}

質問:
{user_query}
"""

    response = model.generate_content(prompt)
    return response.text
