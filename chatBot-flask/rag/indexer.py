# rag/indexer.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 文ベクトル化モデルを1回だけロード
model = SentenceTransformer('all-MiniLM-L6-v2')

def build_faiss_index(text, chunk_size=300):
    # 一定長で分割
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    # ベクトル化
    embeddings = model.encode(chunks)
    # FAISSインデックス作成
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks
