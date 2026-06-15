"""BM25 による語彙ベース検索

Chroma に入っている全チャンクを取り出し、BM25 索引をメモリ上に構築する。
固有名詞・専門用語の一致に強く、ベクトル検索（意味の近さ）を補う。
チャンクの実体は Chroma を単一ソースとし、ここでは複製しない。
"""
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from .tokenizer import tokenize


class BM25Retriever:
    """全チャンクから BM25 索引を作り、語彙一致で検索する"""

    def __init__(self, retriever):
        """
        Args:
            retriever: 構築済みの Chroma Retriever（チャンクの取得元）
        """
        collection = retriever.collection or retriever.get_collection()
        data = collection.get(include=["documents", "metadatas"])
        self.ids = data["ids"]
        self.documents = data["documents"]
        self.metadatas = data["metadatas"]
        self._bm25 = BM25Okapi([tokenize(doc) for doc in self.documents])

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """語彙一致で上位 K 件を返す

        Args:
            query: 質問文
            top_k: 取得件数

        Returns:
            [{id, text, metadata, score}] のリスト（スコア降順）
        """
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "id": self.ids[i],
                "text": self.documents[i],
                "metadata": self.metadatas[i],
                "score": float(scores[i]),
            }
            for i in ranked
        ]
