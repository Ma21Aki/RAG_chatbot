// ベクトル検索と BM25 を RRF で統合し、係名一致で再ランキングする。
// Python版 hybrid_retriever.py 相当（用語注入＋RRF＋係名ブースト）。

export function createHybrid({
  embedder,
  vectorStore,
  bm25,
  tokenizer,
  glossary,
  kakariTerms,
  rrfK = 60,
  boostWeight = 0.03,
}) {
  const chunks = vectorStore.chunks;

  function bm25HitFrom(index, score) {
    const c = chunks[index];
    return {
      id: c.id,
      text: c.text,
      metadata: { source: c.source, folder: c.folder, page: c.page, chunkIdx: c.chunkIdx },
      score,
    };
  }

  async function search(query, { topK = 10, candidateK = 30 } = {}) {
    // 用語注入: 内輪語の定義をクエリへ連結
    const retrievalQuery = glossary ? glossary.expand(query).query : query;

    const queryVec = await embedder.embedSingle(retrievalQuery);
    const vectorHits = vectorStore.search(queryVec, candidateK);
    const queryTokens = tokenizer.tokenize(retrievalQuery);
    const lexicalHits = bm25.search(queryTokens, candidateK).map((h) => bm25HitFrom(h.index, h.score));

    // RRF: 各ランキングで rank（1始まり）に応じた寄与 1/(k+rank) を加算
    const rrf = new Map();
    const info = new Map();
    for (const ranking of [vectorHits, lexicalHits]) {
      ranking.forEach((hit, i) => {
        const rank = i + 1;
        rrf.set(hit.id, (rrf.get(hit.id) || 0) + 1 / (rrfK + rank));
        if (!info.has(hit.id)) info.set(hit.id, hit);
      });
    }

    // 係名ブースト。係名の検出は「元の質問」で行う（拡張後の定義文に混じる別の係名で
    // 誤って加点しないため）。
    const queryTerms = kakariTerms.filter((t) => query.includes(t));
    for (const [id, hit] of info) {
      const source = hit.metadata?.source || '';
      if (queryTerms.length && queryTerms.some((t) => source.includes(t))) {
        rrf.set(id, rrf.get(id) + boostWeight);
        hit.boosted = true;
      }
    }

    return [...rrf.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK)
      .map(([id, score]) => ({ ...info.get(id), rrfScore: score }));
  }

  return { search };
}
