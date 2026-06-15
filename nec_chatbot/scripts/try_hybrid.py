"""hybrid 検索（embedding + BM25 / RRF）の動作確認・比較スクリプト

使い方:
    python nec_chatbot/scripts/try_hybrid.py "総リーのしごとは"

ベクトル単体・BM25 単体・hybrid(RRF) の上位結果を並べて表示し、効果を見比べる。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.rag.embedder import Embedder
from app.rag.retriever import Retriever
from app.rag.bm25_retriever import BM25Retriever
from app.rag.hybrid_retriever import HybridRetriever
from app.config import config


def _label(hit: dict) -> str:
    meta = hit.get("metadata", {})
    return f'{meta.get("source", "?")} (p.{meta.get("page", "?")}#{meta.get("chunk_idx", "?")})'


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "総リーのしごとは"
    top_k = 5

    embedder = Embedder(model_name=config.EMBEDDING_MODEL_NAME)
    retriever = Retriever(db_path=config.CHROMA_DB_PATH)
    retriever.get_collection()
    bm25 = BM25Retriever(retriever)
    hybrid = HybridRetriever(embedder, retriever, bm25)

    print(f"クエリ: {query}\n")

    print("=== ベクトル単体 ===")
    for i, h in enumerate(retriever.search(embedder.embed_single(query), top_k=top_k), 1):
        print(f'{i}. {_label(h)}  dist={h.get("distance"):.3f}')

    print("\n=== BM25単体 ===")
    for i, h in enumerate(bm25.search(query, top_k=top_k), 1):
        print(f'{i}. {_label(h)}  score={h["score"]:.3f}')

    print("\n=== hybrid (RRF) ===")
    for i, h in enumerate(hybrid.search(query, top_k=top_k), 1):
        print(f'{i}. {_label(h)}  rrf={h["rrf_score"]:.4f}')


if __name__ == "__main__":
    main()
