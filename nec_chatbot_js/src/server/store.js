// 利用統計（月別）と不具合報告ログの永続化。
// 低トラフィック前提で、統計は JSON 全体の読み書き、報告は JSONL 追記とする。
import fs from 'node:fs/promises';

// ローカル時刻での「今月」キー（例: 2026-06）
function monthKey(date = new Date()) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  return `${y}-${m}`;
}

export function createStore({ statsPath, feedbackPath }) {
  async function readStats() {
    try {
      return JSON.parse(await fs.readFile(statsPath, 'utf8'));
    } catch {
      return {}; // 未作成・破損時は空集計から始める
    }
  }

  // 1回の質問応答ぶんの利用量を今月に加算
  async function recordUsage({ input = 0, output = 0 } = {}) {
    const stats = await readStats();
    const key = monthKey();
    const cur = stats[key] || { count: 0, inputTokens: 0, outputTokens: 0 };
    cur.count += 1;
    cur.inputTokens += input;
    cur.outputTokens += output;
    stats[key] = cur;
    await fs.writeFile(statsPath, JSON.stringify(stats, null, 2));
  }

  // 今月の統計（未記録なら 0 を返す）
  async function getMonthlyStats() {
    const stats = await readStats();
    const key = monthKey();
    const cur = stats[key] || { count: 0, inputTokens: 0, outputTokens: 0 };
    return { month: key, ...cur };
  }

  // 不具合報告を1行追記
  async function appendFeedback(entry) {
    const line = JSON.stringify({ time: new Date().toISOString(), ...entry }) + '\n';
    await fs.appendFile(feedbackPath, line);
  }

  return { recordUsage, getMonthlyStats, appendFeedback };
}
