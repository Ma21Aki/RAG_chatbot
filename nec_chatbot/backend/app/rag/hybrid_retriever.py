"""ベクトル検索と BM25 を Reciprocal Rank Fusion (RRF) で統合する

RRF は各リトリーバの「順位」だけを使って融合するため、コサイン距離と BM25 スコアの
ように尺度が異なるランキングでも、正規化なしで安全に統合できる。

検索の前後に2つの工夫を入れる。
- 用語注入（glossary）: 質問に含まれる内輪語の定義をクエリへ連結し、埋め込み側が
  内輪語を理解できない問題を緩和する。
- ファイル名一致ブースト: 質問に含まれる係名（kakari_terms.csv）がチャンクの出典
  ファイル名に一致したら加点し、その係の文書を上位へ押し上げる。
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
    """embedding 検索と BM25 検索を RRF で融合し、係名一致で再ランキングする"""

    def __init__(self, embedder, retriever, bm25_retriever, rrf_k: int = 60,
                 boost_weight: float = 0.03, glossary=None):
        """
        Args:
            embedder: 質問のベクトル化
            retriever: Chroma ベクトル検索
            bm25_retriever: BM25 語彙検索
            rrf_k: RRF の定数。大きいほど上位・下位の差が緩やかになる（慣例値 60）
            boost_weight: ファイル名一致ブーストの加点量。RRF 寄与（約 0.016〜0.032）に
                対する加点なので 0.03 前後が目安
            glossary: 用語注入用の Glossary。None なら用語注入は行わない
        """
        self.embedder = embedder
        self.retriever = retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k
        self.boost_weight = boost_weight
        self.glossary = glossary

    def search(self, query: str, top_k: int = 10, candidate_k: int = 30) -> List[Dict[str, Any]]:
        """用語注入＋RRF統合＋係名ブーストの検索結果を返す

        Args:
            query: 質問文
            top_k: 最終的に返す件数
            candidate_k: 各リトリーバから集める候補数（融合・加点はこの範囲で効く）

        Returns:
            [{id, text, metadata, ..., rrf_score, boosted}] のリスト（スコア降順）
        """
        # 用語注入: 内輪語の定義をクエリへ連結（検索の手がかりを増やす）
        retrieval_query = query
        if self.glossary is not None:
            retrieval_query, _ = self.glossary.expand(query)

        query_embedding = self.embedder.embed_single(retrieval_query)
        vector_hits = self.retriever.search(query_embedding, top_k=candidate_k)
        lexical_hits = self.bm25.search(retrieval_query, top_k=candidate_k)

        rrf_scores: Dict[str, float] = {}
        hit_info: Dict[str, Dict[str, Any]] = {}

        # 各ランキングで rank（1始まり）に応じた寄与 1/(k+rank) を加算
        for ranking in (vector_hits, lexical_hits):
            for rank, hit in enumerate(ranking, start=1):
                doc_id = hit["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
                hit_info.setdefault(doc_id, hit)

        # ファイル名一致ブースト。係名の検出は「元の質問」で行う（拡張後の定義文には
        # 別の係名が混じりうるため、それで加点すると誤って他の係を持ち上げてしまう）
        query_terms = [t for t in _KAKARI_TERMS if t in query]
        for doc_id, hit in hit_info.items():
            source = hit.get("metadata", {}).get("source", "")
            if query_terms and any(t in source for t in query_terms):
                rrf_scores[doc_id] += self.boost_weight
                hit["boosted"] = True

        ranked_ids = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:top_k]
        return [{**hit_info[i], "rrf_score": rrf_scores[i]} for i in ranked_ids]
