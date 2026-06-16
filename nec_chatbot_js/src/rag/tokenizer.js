// 日本語の分かち書き（kuromoji + 用語保護層）
// Python版 tokenizer.py（Janome + ユーザー辞書）の置換。
// kuromoji は内蔵IPADICのため固有語が誤分割される。protect_terms の語は
// トークナイズ前に切り出して1語として扱う「用語保護層」で守る。
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import kuromoji from 'kuromoji';

const require = createRequire(import.meta.url);

// 検索の手がかりになる内容語だけ残す（助詞・記号などは捨てる）
const KEEP_POS = new Set(['名詞', '動詞', '形容詞', '副詞']);

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function loadTermList(text) {
  return text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith('#'));
}

function buildKuromoji() {
  const dicPath = path.join(path.dirname(require.resolve('kuromoji')), '..', 'dict');
  return new Promise((resolve, reject) => {
    kuromoji.builder({ dicPath }).build((err, tokenizer) => {
      if (err) reject(err);
      else resolve(tokenizer);
    });
  });
}

export async function createTokenizer({ protectTermsPath }) {
  const kuro = await buildKuromoji();

  // 用語保護層: 長い語を優先一致させるため文字数降順で並べる
  const protectTerms = loadTermList(await readFile(protectTermsPath, 'utf-8'))
    .sort((a, b) => b.length - a.length);
  const protectSet = new Set(protectTerms);
  const protectRegex = protectTerms.length
    ? new RegExp('(' + protectTerms.map(escapeRegExp).join('|') + ')', 'g')
    : null;

  // kuromoji で分かち書きし、内容語の原形（小文字）を返す
  function kuromojiTokens(segment) {
    const out = [];
    for (const t of kuro.tokenize(segment)) {
      if (!KEEP_POS.has(t.pos)) continue;
      const base = t.basic_form && t.basic_form !== '*' ? t.basic_form : t.surface_form;
      out.push(base.toLowerCase());
    }
    return out;
  }

  function tokenize(text) {
    if (!protectRegex) return kuromojiTokens(text);
    const tokens = [];
    // 保護語で分割。capturing split により保護語自身も配列に残る
    for (const part of text.split(protectRegex)) {
      if (!part) continue;
      if (protectSet.has(part)) tokens.push(part.toLowerCase()); // 固有語は1語として確定
      else tokens.push(...kuromojiTokens(part));
    }
    return tokens;
  }

  return { tokenize };
}
