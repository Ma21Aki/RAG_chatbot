// RAG パイプライン（Python版 pipeline.py 相当）
// 検索（hybrid）→ 文脈整形 → 生成 → 出典抽出 を束ねる。
import { SYSTEM_PROMPT, formatContext, formatSources } from './contextFormatter.js';

export function createPipeline({ hybrid, generator }) {
  async function ask(query, { topK } = {}) {
    const t0 = Date.now();
    const searchResults = await hybrid.search(query, { topK });
    const tSearch = Date.now();

    const context = formatContext(searchResults);
    const gen = await generator.generate({ system: SYSTEM_PROMPT, context, query });
    const tGen = Date.now();

    return {
      query,
      answer: gen.text,
      sources: formatSources(searchResults),
      searchResults,
      metrics: {
        searchTime: (tSearch - t0) / 1000,
        generationTime: (tGen - tSearch) / 1000,
        totalTime: (tGen - t0) / 1000,
        tokens: gen.usage,
        numResults: searchResults.length,
      },
    };
  }

  return { ask };
}
