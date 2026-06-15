"""ベクトル検索と BM25 を Reciprocal Rank Fusion (RRF) で統合する

RRF は各リトリーバの「順位」だけを使って融合するため、コサイン距離と BM25 スコアの
ように尺度が異なるランキングでも、正規化なしで安全に統合できる。
"""
from typing import List, Dict, Any


class HybridRetriever:
    """embedding 検索と BM25 検索の順位を RRF で融合する"""

    def __init__(self, embedder, retriever, bm25_retriever, rrf_k: int = 60):
        """
        Args:
            embedder: 質問のベクトル化
            retriever: Chroma ベクトル検索
            bm25_retriever: BM25 語彙検索
            rrf_k: RRF の定数。大きいほど上位・下位の差が緩やかになる（慣例値 60）
        """
        self.embedder = embedder
        self.retriever = retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10, candidate_k: int = 20,
               query_embedding: List[float] = None) -> List[Dict[str, Any]]:
        """RRF で統合した検索結果を返す

        Args:
            query: 質問文
            top_k: 最終的に返す件数
            candidate_k: 各リトリーバから集める候補数（融合前）
            query_embedding: 事前計算済みの質問ベクトル。渡さなければ内部で計算する
                （pipeline 側で埋め込み時間を計測したい場合に渡す）

        Returns:
            [{id, text, metadata, ..., rrf_score}] のリスト（RRF スコア降順）
        """
        if query_embedding is None:
            query_embedding = self.embedder.embed_single(query)
        vector_hits = self.retriever.search(query_embedding, top_k=candidate_k)
        lexical_hits = self.bm25.search(query, top_k=candidate_k)

        rrf_scores: Dict[str, float] = {}
        hit_info: Dict[str, Dict[str, Any]] = {}

        # 各ランキングで rank（1始まり）に応じた寄与 1/(k+rank) を加算
        for ranking in (vector_hits, lexical_hits):
            for rank, hit in enumerate(ranking, start=1):
                doc_id = hit["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
                hit_info.setdefault(doc_id, hit)

        ranked_ids = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:top_k]
        return [{**hit_info[i], "rrf_score": rrf_scores[i]} for i in ranked_ids]
