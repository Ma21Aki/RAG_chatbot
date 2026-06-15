"""ベクトル検索と BM25 を Reciprocal Rank Fusion (RRF) で統合する

RRF は各リトリーバの「順位」だけを使って融合するため、コサイン距離と BM25 スコアの
ように尺度が異なるランキングでも、正規化なしで安全に統合できる。

さらに、融合後に「ファイル名一致ブースト」を行う。質問に含まれる係名（kakari_terms.csv）
がチャンクの出典ファイル名に一致したら加点し、その係の文書を上位へ押し上げる。
「係の質問なのに別の係が上位に来る」問題への直接対策。
"""
from pathlib import Path
from typing import List, Dict, Any

_KAKARI_TERMS_PATH = Path(__file__).parent / "kakari_terms.csv"


def _load_kakari_terms(path: Path) -> List[str]:
    """係名リストを読み込む（空行と # コメントは無視）"""
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


_KAKARI_TERMS = _load_kakari_terms(_KAKARI_TERMS_PATH)


class HybridRetriever:
    """embedding 検索と BM25 検索の順位を RRF で融合し、係名一致で再ランキングする"""

    def __init__(self, embedder, retriever, bm25_retriever, rrf_k: int = 60,
                 boost_weight: float = 0.03):
        """
        Args:
            embedder: 質問のベクトル化
            retriever: Chroma ベクトル検索
            bm25_retriever: BM25 語彙検索
            rrf_k: RRF の定数。大きいほど上位・下位の差が緩やかになる（慣例値 60）
            boost_weight: ファイル名一致ブーストの加点量。RRF 寄与（約 0.016〜0.032）に
                対する加点なので 0.03 前後が目安。大きすぎると常に係名一致が最優先になる
        """
        self.embedder = embedder
        self.retriever = retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k
        self.boost_weight = boost_weight

    def search(self, query: str, top_k: int = 10, candidate_k: int = 30,
               query_embedding: List[float] = None) -> List[Dict[str, Any]]:
        """RRF 統合＋係名一致ブーストの検索結果を返す

        Args:
            query: 質問文
            top_k: 最終的に返す件数
            candidate_k: 各リトリーバから集める候補数（融合・加点はこの範囲で効く）
            query_embedding: 事前計算済みの質問ベクトル。渡さなければ内部で計算する
                （pipeline 側で埋め込み時間を計測したい場合に渡す）

        Returns:
            [{id, text, metadata, ..., rrf_score, boosted}] のリスト（スコア降順）
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

        # ファイル名一致ブースト: 質問に含まれる係名が出典ファイル名にも含まれれば加点
        query_terms = [t for t in _KAKARI_TERMS if t in query]
        for doc_id, hit in hit_info.items():
            source = hit.get("metadata", {}).get("source", "")
            if query_terms and any(t in source for t in query_terms):
                rrf_scores[doc_id] += self.boost_weight
                hit["boosted"] = True

        ranked_ids = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:top_k]
        return [{**hit_info[i], "rrf_score": rrf_scores[i]} for i in ranked_ids]
