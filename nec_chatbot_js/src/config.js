// 設定（.env を読み込み、パスと各種パラメータを集約）
import 'dotenv/config';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..'); // src/ の一つ上＝プロジェクトルート

export const config = {
  root,
  docsDir: path.join(root, 'data', 'docs'),
  indexPath: path.join(root, 'data', 'index.json'),
  statsPath: path.join(root, 'data', 'stats.json'), // 月別の利用統計
  feedbackPath: path.join(root, 'data', 'feedback.jsonl'), // 不具合報告ログ（1行1件）
  dictDir: path.join(root, 'data', 'dict'),
  glossaryPath: path.join(root, 'data', 'dict', 'glossary.md'),
  kakariTermsPath: path.join(root, 'data', 'dict', 'kakari_terms.txt'),
  protectTermsPath: path.join(root, 'data', 'dict', 'protect_terms.txt'),
  modelsDir: path.join(root, 'models'),

  // 埋め込み
  embeddingModel: process.env.EMBEDDING_MODEL || 'Xenova/paraphrase-multilingual-MiniLM-L12-v2',

  // 検索（実験で調整するパラメータ）
  topK: Number(process.env.TOP_K || 10),
  candidateK: Number(process.env.CANDIDATE_K || 30),

  // 検索モード別の取得件数（UIの「検索設定」で切り替え）
  searchModes: {
    broad: Number(process.env.SEARCH_K_BROAD || 4), // ざっくり検索（デフォルト・少なめに絞る）
    detailed: Number(process.env.SEARCH_K_DETAILED || 10), // 詳しく検索（幅広く見る）
  },
  defaultSearchMode: 'broad',

  // 生成（Gemini）
  gemini: {
    apiKey: process.env.GEMINI_API_KEY || '',
    model: process.env.GEMINI_MODEL || 'gemini-2.5-flash',
    maxOutputTokens: Number(process.env.GEMINI_MAX_OUTPUT_TOKENS || 500),
    temperature: Number(process.env.GEMINI_TEMPERATURE || 0.3),
  },

  // サーバー
  server: {
    port: Number(process.env.PORT || 3000),
    authToken: process.env.AUTH_TOKEN || '',
    rateLimit: {
      windowMs: Number(process.env.RATE_LIMIT_WINDOW_MS || 60000),
      max: Number(process.env.RATE_LIMIT_MAX || 30),
    },
  },
};
