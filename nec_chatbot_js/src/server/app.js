// Express アプリ（静的UI配信 ＋ /api）
import express from 'express';
import path from 'node:path';
import { config } from '../config.js';
import { authMiddleware } from './auth.js';
import { rateLimiter } from './rateLimit.js';

const publicDir = path.join(config.root, 'public');
const MAX_QUERY_LEN = 500;

export function createApp({ pipeline, hybrid }) {
  const app = express();
  app.set('trust proxy', true); // Tunnel/プロキシ経由でも req.ip を正しく取る
  app.use(express.json({ limit: '32kb' }));
  app.use(express.static(publicDir));

  // 入力検証（質問の取り出し）
  function getQuery(req, res) {
    const q = (req.body?.query ?? '').toString().trim();
    if (!q) {
      res.status(400).json({ error: '質問を入力してください' });
      return null;
    }
    if (q.length > MAX_QUERY_LEN) {
      res.status(400).json({ error: '質問が長すぎます' });
      return null;
    }
    return q;
  }

  const api = express.Router();
  api.use(authMiddleware(config.server.authToken));
  api.use(rateLimiter(config.server.rateLimit));

  // 質問応答（検索＋生成）
  api.post('/ask', async (req, res) => {
    const q = getQuery(req, res);
    if (!q) return;
    try {
      const r = await pipeline.ask(q, {});
      res.json({ answer: r.answer, sources: r.sources, metrics: r.metrics });
    } catch (e) {
      console.error('[ask]', e);
      res.status(500).json({ error: '回答の生成に失敗しました' });
    }
  });

  // 検索のみ（LLM不要・デバッグ/透明性）
  api.post('/search', async (req, res) => {
    const q = getQuery(req, res);
    if (!q) return;
    try {
      const results = await hybrid.search(q, {});
      res.json({
        results: results.map((h) => ({
          source: h.metadata.source,
          folder: h.metadata.folder,
          page: h.metadata.page,
          rrfScore: h.rrfScore,
          boosted: !!h.boosted,
          text: h.text,
        })),
      });
    } catch (e) {
      console.error('[search]', e);
      res.status(500).json({ error: '検索に失敗しました' });
    }
  });

  app.use('/api', api);
  return app;
}
