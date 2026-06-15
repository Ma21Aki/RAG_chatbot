"""検索の定量評価（質問→期待出典）

ベクトル / BM25 / hybrid(ブーストなし) / hybrid(ブーストあり) の4方式で、
Hit@k（期待する出典が上位 k に入った割合）と MRR（最初に当たった順位の逆数の平均）を比較する。
各施策（BM25・RRF・係名ブースト）が数字でどれだけ効いたかを見るためのもの。

評価データ: scripts/eval_set.json  … [{query, expected: [出典の部分文字列, ...]}]
  expected はチャンクの metadata["source"] に対する部分一致。いずれか1つ当たれば hit。
使い方: python nec_chatbot/scripts/evaluate.py [top_k]
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.rag.embedder import Embedder
from app.rag.retriever import Retriever
from app.rag.bm25_retriever import BM25Retriever
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.glossary import Glossary
from app.config import config

EVAL_PATH = Path(__file__).parent / "eval_set.json"


def _hit_rank(results: list, expected: list) -> int:
    """expected のいずれかに最初に一致した順位（1始まり）。無ければ 0"""
    for rank, r in enumerate(results, 1):
        source = r.get("metadata", {}).get("source", "")
        if any(e in source for e in expected):
            return rank
    return 0


def _evaluate(name: str, search_fn, cases: list, top_k: int) -> list:
    hits = 0
    rr_sum = 0.0
    misses = []
    for case in cases:
        results = search_fn(case["query"])[:top_k]
        rank = _hit_rank(results, case["expected"])
        if rank:
            hits += 1
            rr_sum += 1.0 / rank
        else:
            misses.append(case["query"])
    n = len(cases)
    print(f"{name:20s} Hit@{top_k}={hits}/{n} ({hits / n:.0%})  MRR={rr_sum / n:.3f}")
    return misses


def main():
    top_k = int(sys.argv[1]) if len(sys.argv) > 1 else config.CHROMA_TOP_K
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    embedder = Embedder(model_name=config.EMBEDDING_MODEL_NAME)
    retriever = Retriever(db_path=config.CHROMA_DB_PATH)
    retriever.get_collection()
    bm25 = BM25Retriever(retriever)
    glossary = Glossary(config.PDF_DATA_DIR)
    hybrid_boost = HybridRetriever(embedder, retriever, bm25)  # 係名ブーストのみ
    hybrid_full = HybridRetriever(embedder, retriever, bm25, glossary=glossary)  # ブースト＋用語注入

    methods = {
        "vector": lambda q: retriever.search(embedder.embed_single(q), top_k=top_k),
        "bm25": lambda q: bm25.search(q, top_k=top_k),
        "hybrid(ブースト)": lambda q: hybrid_boost.search(q, top_k=top_k),
        "hybrid(ブースト+用語注入)": lambda q: hybrid_full.search(q, top_k=top_k),
    }

    print(f"評価: {len(cases)}問, top_k={top_k}\n")
    misses = {}
    for name, fn in methods.items():
        misses[name] = _evaluate(name, fn, cases, top_k)

    print("\n--- hybrid(ブースト+用語注入) で外した質問 ---")
    print("\n".join(misses["hybrid(ブースト+用語注入)"]) or "なし")


if __name__ == "__main__":
    main()
