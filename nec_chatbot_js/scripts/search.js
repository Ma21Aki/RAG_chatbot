// 検索・回答のCLI確認（Python版 try_hybrid / ask_to_chatbot 相当）
// 使い方:
//   node scripts/search.js "総リーの仕事は"          検索のみ（LLM不要）
//   node scripts/search.js --ask "総リーの仕事は"     生成まで（要 GEMINI_API_KEY）
import { bootstrap } from '../src/rag/bootstrap.js';

const args = process.argv.slice(2);
const ask = args.includes('--ask');
const query = args.filter((a) => a !== '--ask').join(' ') || '総リーの仕事は？';

const { hybrid, pipeline } = await bootstrap({ withGenerator: ask });

const results = await hybrid.search(query, {});
console.log(`クエリ: ${query}\n`);
console.log('=== 検索結果 ===');
results.forEach((h, i) => {
  const m = h.metadata;
  const mark = h.boosted ? ' [boost]' : '';
  console.log(`${i + 1}. ${m.source} (p.${m.page}#${m.chunkIdx})  rrf=${h.rrfScore.toFixed(4)}${mark}`);
});

if (ask) {
  const r = await pipeline.ask(query, {});
  console.log('\n=== 回答 ===');
  console.log(r.answer);
  console.log('\n=== 出典 ===');
  r.sources.forEach((s) => console.log(`  - ${s.source} (p.${s.page})`));
  console.log('\n=== メトリクス ===');
  console.log(
    `  search=${r.metrics.searchTime}s gen=${r.metrics.generationTime}s total=${r.metrics.totalTime}s tokens=${r.metrics.tokens.total}`
  );
}
// process.exit は呼ばない（Windowsでの終了時アサーション回避。イベントループが空けば自然終了）
