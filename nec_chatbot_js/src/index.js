// サーバー起動エントリ。コンポーネントを構築して Express を起動する。
import { bootstrap } from './rag/bootstrap.js';
import { createApp } from './server/app.js';
import { config } from './config.js';

console.log('[INIT] コンポーネント構築中（モデル読み込み）...');
const { pipeline, hybrid } = await bootstrap({ withGenerator: true });

const app = createApp({ pipeline, hybrid });
app.listen(config.server.port, () => {
  console.log(`[READY] http://localhost:${config.server.port}`);
  if (!config.gemini.apiKey) console.log('[WARN] GEMINI_API_KEY 未設定。/api/ask は失敗します（/api/search は可）');
  if (!config.server.authToken) console.log('[WARN] AUTH_TOKEN 未設定。認証は無効（ローカル用）');
});
