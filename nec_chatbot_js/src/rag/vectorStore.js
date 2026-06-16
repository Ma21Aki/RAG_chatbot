// ベクトル検索（index.json を読み込み、総当りコサイン類似度）
// Python版 retriever.py（Chroma）の置換。数百チャンク規模ではDB不要。
import { readFile } from 'node:fs/promises';

export async function loadVectorStore(indexPath) {
  const data = JSON.parse(await readFile(indexPath, 'utf-8'));
  const chunks = data.chunks;

  // 埋め込みは正規化済み（embed時 normalize:true）なので、内積＝コサイン類似度。
  function search(queryVec, k) {
    const scored = chunks.map((c) => {
      const e = c.embedding;
      let dot = 0;
      for (let i = 0; i < e.length; i++) dot += e[i] * queryVec[i];
      return { chunk: c, score: dot };
    });
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k).map(({ chunk, score }) => ({
      id: chunk.id,
      text: chunk.text,
      metadata: {
        source: chunk.source,
        folder: chunk.folder,
        page: chunk.page,
        chunkIdx: chunk.chunkIdx,
      },
      distance: 1 - score,
    }));
  }

  return { chunks, meta: { model: data.model, dim: data.dim }, search };
}
