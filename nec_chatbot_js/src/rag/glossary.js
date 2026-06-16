// 用語注入（クエリ拡張）。Python版 glossary.py 相当。
// 用語辞典(.md)を ## 見出し単位でパースし、質問に出た語の定義をクエリへ連結する。
import { readFile } from 'node:fs/promises';

const MIN_TERM_LEN = 2; // 1文字の語は誤爆しやすいので対象外

export async function loadGlossary(glossaryPath) {
  const text = await readFile(glossaryPath, 'utf-8');
  const terms = new Map(); // 別名 -> 定義

  let heading = null;
  let body = [];
  const flush = () => {
    if (heading && body.length) {
      const def = body.join(' ').trim();
      for (let alias of heading.split(/[・、]/)) {
        alias = alias.trim();
        if (alias.length >= MIN_TERM_LEN && def) terms.set(alias, def);
      }
    }
  };

  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith('##')) {
      flush();
      heading = line.replace(/^#+/, '').trim();
      body = [];
    } else if (heading) {
      body.push(line.trim());
    }
  }
  flush();

  // 質問に出現した用語の定義を連結した検索クエリを返す
  function expand(query) {
    const seen = new Set();
    const defs = [];
    const matched = [];
    for (const [alias, def] of terms) {
      if (query.includes(alias) && !seen.has(def)) {
        seen.add(def);
        defs.push(def);
        matched.push(alias);
      }
    }
    if (!defs.length) return { query, matched: [] };
    return { query: query + ' ' + defs.join(' '), matched };
  }

  return { terms, expand };
}
