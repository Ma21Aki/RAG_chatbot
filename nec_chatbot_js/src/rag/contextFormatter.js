// プロンプト整形（Python版 prompt_templates.py 相当）
// ① 同一文書まとめ＋文書順 ② 隣接チャンク結合 ③ ヘッダー重複整理

export const SYSTEM_PROMPT = `あなたはサークルのマニュアル・書類に基づいた質問応答システムです。
以下のルールに従ってください：

1. 提供された【参考情報】の範囲内で回答してください
2. 参考情報に該当する内容がない場合は、「マニュアルに記載されていません」と答えてください
3. 不確実な情報は含めないでください
4. 回答は簡潔で分かりやすくしてください
5. 必要に応じて箇条書きで整理してください
`;

// 本文先頭の【係名】行を除く（出典はグループ見出しで一度だけ示すため）
function stripHeader(text) {
  return text.replace(/^【[^】]*】\n?/, '');
}

// a の末尾と b の先頭が重なっていればその分を除いて連結（重複削減）。無ければ改行連結。
function joinOverlap(a, b, maxOverlap = 200) {
  const limit = Math.min(a.length, b.length, maxOverlap);
  for (let k = limit; k > 0; k--) {
    if (a.slice(-k) === b.slice(0, k)) return a + b.slice(k);
  }
  return a + '\n' + b;
}

export function formatContext(results) {
  if (!results.length) return '（該当する情報がありません）';

  // ① 同一文書でグループ化（出現順＝関連度順を保持）
  const groups = [];
  const index = new Map();
  for (const r of results) {
    const m = r.metadata || {};
    const key = `${m.folder || ''}|${m.source || 'Unknown'}`;
    if (!index.has(key)) {
      index.set(key, groups.length);
      groups.push({ folder: m.folder || '', source: m.source || 'Unknown', items: [] });
    }
    groups[index.get(key)].items.push(r);
  }

  const blocks = [];
  groups.forEach((g, gi) => {
    const sourceDisplay = g.folder && g.folder !== '.' ? `${g.folder}/${g.source}` : g.source;

    // 文書内はページ・chunkIdx 順
    g.items.sort(
      (a, b) => a.metadata.page - b.metadata.page || a.metadata.chunkIdx - b.metadata.chunkIdx
    );

    // ② 連続チャンク（同一ページ・連番）を結合
    const segments = [];
    for (const r of g.items) {
      const page = r.metadata.page;
      const idx = r.metadata.chunkIdx;
      const body = stripHeader(r.text || '').trim(); // ③ ヘッダー除去
      const last = segments[segments.length - 1];
      if (last && last.page === page && idx === last.idxEnd + 1) {
        last.text = joinOverlap(last.text, body);
        last.idxEnd = idx;
      } else {
        segments.push({ page, idxEnd: idx, text: body });
      }
    }

    const lines = [`[${gi + 1}] ${sourceDisplay}`];
    for (const seg of segments) lines.push(`(p.${seg.page}) ${seg.text}`);
    blocks.push(lines.join('\n'));
  });

  return blocks.join('\n\n');
}

export function formatSources(results) {
  const sources = [];
  const seen = new Set();
  for (const r of results) {
    const m = r.metadata || {};
    const key = `${m.source}|${m.folder}|${m.page}`;
    if (!seen.has(key)) {
      seen.add(key);
      sources.push({ source: m.source, folder: m.folder, page: m.page });
    }
  }
  return sources;
}
