// BM25 による語彙ベース検索（Python版 bm25_retriever.py 相当）
// 分かち書き済みの文書集合から索引を作り、クエリトークンでスコアリングする。

export function buildBm25(docsTokens, { k1 = 1.5, b = 0.75 } = {}) {
  const N = docsTokens.length;
  const docLen = docsTokens.map((t) => t.length);
  const avgdl = N ? docLen.reduce((a, c) => a + c, 0) / N : 0;

  // 文書ごとの語頻度
  const tf = docsTokens.map((tokens) => {
    const m = new Map();
    for (const t of tokens) m.set(t, (m.get(t) || 0) + 1);
    return m;
  });

  // 文書頻度 → IDF（負値を避ける log(1 + ...) 形）
  const df = new Map();
  for (const m of tf) for (const t of m.keys()) df.set(t, (df.get(t) || 0) + 1);
  const idf = new Map();
  for (const [t, n] of df) idf.set(t, Math.log(1 + (N - n + 0.5) / (n + 0.5)));

  function score(queryTokens, i) {
    const m = tf[i];
    const dl = docLen[i];
    let s = 0;
    for (const t of queryTokens) {
      const f = m.get(t);
      if (!f) continue;
      const id = idf.get(t) || 0;
      s += id * (f * (k1 + 1)) / (f + k1 * (1 - b + (b * dl) / (avgdl || 1)));
    }
    return s;
  }

  function search(queryTokens, k) {
    const q = [...new Set(queryTokens)];
    const scored = [];
    for (let i = 0; i < N; i++) scored.push({ index: i, score: score(q, i) });
    scored.sort((a, c) => c.score - a.score);
    return scored.slice(0, k);
  }

  return { search };
}
