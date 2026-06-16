// 各コンポーネントを構築して pipeline まで組み上げる（CLI・サーバー共通の入口）
import { readFile } from 'node:fs/promises';
import { config } from '../config.js';
import { createEmbedder } from './embedder.js';
import { loadVectorStore } from './vectorStore.js';
import { createTokenizer } from './tokenizer.js';
import { buildBm25 } from './bm25.js';
import { loadGlossary } from './glossary.js';
import { createHybrid } from './hybrid.js';
import { createGenerator } from './generator/index.js';
import { createPipeline } from './pipeline.js';

function loadTermList(text) {
  return text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith('#'));
}

// withGenerator=false なら検索のみ（Geminiキー不要）で構築する
export async function bootstrap({ withGenerator = true } = {}) {
  const embedder = await createEmbedder({ model: config.embeddingModel });
  const vectorStore = await loadVectorStore(config.indexPath);
  const tokenizer = await createTokenizer({ protectTermsPath: config.protectTermsPath });

  // 索引済みチャンクから BM25 を構築（チャンクの単一ソースは index.json）
  const docsTokens = vectorStore.chunks.map((c) => tokenizer.tokenize(c.text));
  const bm25 = buildBm25(docsTokens);

  const glossary = await loadGlossary(config.glossaryPath);
  const kakariTerms = loadTermList(await readFile(config.kakariTermsPath, 'utf-8'));

  const hybrid = createHybrid({
    embedder,
    vectorStore,
    bm25,
    tokenizer,
    glossary,
    kakariTerms,
    rrfK: 60,
    boostWeight: 0.03,
  });

  const generator = withGenerator ? createGenerator(config) : null;
  const pipeline = generator ? createPipeline({ hybrid, generator }) : null;

  return { embedder, vectorStore, tokenizer, bm25, glossary, hybrid, generator, pipeline };
}
