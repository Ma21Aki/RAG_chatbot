// Express アプリ（静的UI配信 ＋ /api）
import express from 'express';
import path from 'node:path';
import { config } from '../config.js';
import { authMiddleware } from './auth.js';
import { rateLimiter } from './rateLimit.js';
import { createStore } from './store.js';

const publicDir = path.join(config.root, 'public');
const MAX_QUERY_LEN = 500;
const FEEDBACK_TYPES = ['wrong', 'missing', 'other']; // 間違った情報 / 必要な情報が出ない / その他
const MAX_FEEDBACK_DETAIL = 2000;

export function createApp({ pipeline, hybrid }) {
  const app = express();
  const store = createStore(config);
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

  // 検索モード→件数の対応（クライアントに任意件数を指定させないため許可リストで決める）
  const MODE_TO_TOPK = config.searchModes;

  // 質問応答（検索＋生成）
  api.post('/ask', async (req, res) => {
    const q = getQuery(req, res);
    if (!q) return;
    const mode = (req.body?.mode ?? '').toString();
    const topK = MODE_TO_TOPK[mode] ?? MODE_TO_TOPK[config.defaultSearchMode];
    try {
      const r = await pipeline.ask(q, { topK });
      // 利用統計に加算（失敗しても回答返却は妨げない）
      store.recordUsage(r.metrics?.tokens).catch((e) => console.error('[stats]', e));
      res.json({ answer: r.answer, sources: r.sources, metrics: r.metrics });
    } catch (e) {
      console.error('[ask]', e);
      res.status(500).json({ error: '回答の生成に失敗しました' });
    }
  });

  // 今月の利用統計（質問回数・入出力トークン）
  api.get('/stats', async (_req, res) => {
    try {
      res.json(await store.getMonthlyStats());
    } catch (e) {
      console.error('[stats]', e);
      res.status(500).json({ error: '統計の取得に失敗しました' });
    }
  });

  // 不具合報告（管理者は data/feedback.jsonl を確認する）
  api.post('/feedback', async (req, res) => {
    const b = req.body || {};
    const type = (b.type ?? '').toString();
    if (!FEEDBACK_TYPES.includes(type)) {
      return res.status(400).json({ error: '不具合の種別が不正です' });
    }
    const entry = {
      type,
      detail: (b.detail ?? '').toString().slice(0, MAX_FEEDBACK_DETAIL),
      query: (b.query ?? '').toString().slice(0, MAX_QUERY_LEN),
      answer: (b.answer ?? '').toString().slice(0, 4000),
    };
    try {
      await store.appendFeedback(entry);
      res.json({ ok: true });
    } catch (e) {
      console.error('[feedback]', e);
      res.status(500).json({ error: '報告の保存に失敗しました' });
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
