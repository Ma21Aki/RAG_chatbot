// 索引構築（Python版 rebuild_index.py 相当）
// data/docs の PDF/MD を取り込み、チャンク分割→ローカル埋め込み→ data/index.json に保存する。
import { readFile, readdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { PDFParse } from 'pdf-parse';
import { config } from '../src/config.js';
import { createEmbedder } from '../src/rag/embedder.js';

// 索引に含めるチャンクの最小文字数（空白を除く）。これ未満は薄いチャンクとして除外。実験で調整。
const MIN_CHUNK_CHARS = 30;
const CHUNK_SIZE = 500;
const OVERLAP = 50;
const EMBED_BATCH = 32;

// --- テキスト整形・チャンク分割（Python版のロジックを移植） ---

function normalizeText(text) {
  const lines = [];
  for (let line of text.split(/\r?\n/)) {
    line = line.replace(/[ \t　]+/g, ' ').trim(); // 行内の連続空白を圧縮
    if (line) lines.push(line);
  }
  return lines.join('\n');
}

function chunkText(text, chunkSize = CHUNK_SIZE, overlap = OVERLAP) {
  if (!text || !text.trim()) return [];
  const chunks = [];
  let start = 0;
  const step = Math.max(chunkSize - overlap, 1);
  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    const chunk = text.slice(start, end);
    if (chunk.trim()) chunks.push(chunk);
    if (end >= text.length) break;
    start += step;
  }
  return chunks.length ? chunks : [text];
}

// Markdown は見出し（# 行）単位で分割し、長い節はさらに固定長で割る
function chunkMarkdown(text, chunkSize = CHUNK_SIZE, overlap = OVERLAP) {
  const sections = [];
  let current = [];
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith('#') && current.length) {
      sections.push(current.join('\n'));
      current = [line];
    } else {
      current.push(line);
    }
  }
  if (current.length) sections.push(current.join('\n'));

  const chunks = [];
  for (let section of sections) {
    section = section.trim();
    if (!section) continue;
    if (section.length <= chunkSize) chunks.push(section);
    else chunks.push(...chunkText(section, chunkSize, overlap));
  }
  return chunks;
}

const meaningfulLen = (s) => s.replace(/\s/g, '').length;

// --- 取り込み ---

// PDF/MD からページ単位のテキストを取り出す（.md は全体を1ページ扱い）
async function extractPages(filePath) {
  if (filePath.toLowerCase().endsWith('.md')) {
    const text = await readFile(filePath, 'utf-8');
    return text.trim() ? [{ text, page: 1 }] : [];
  }
  const buffer = await readFile(filePath);
  const parser = new PDFParse({ data: new Uint8Array(buffer) });
  try {
    const result = await parser.getText();
    return result.pages
      .filter((p) => p.text && p.text.trim())
      .map((p) => ({ text: p.text, page: p.num }));
  } finally {
    if (typeof parser.destroy === 'function') await parser.destroy();
  }
}

// 索引対象の収集。同じ場所に同名 .md がある PDF は抽出失敗版としてスキップ。
async function collectSourceFiles(docsDir) {
  const entries = await readdir(docsDir, { recursive: true, withFileTypes: true });
  const files = entries
    .filter((e) => e.isFile())
    .map((e) => path.join(e.parentPath ?? e.path, e.name))
    .filter((f) => /\.(pdf|md)$/i.test(f));

  const mdKeys = new Set(
    files
      .filter((f) => f.toLowerCase().endsWith('.md'))
      .map((f) => `${path.dirname(f)}|${path.basename(f, path.extname(f))}`)
  );

  return files.filter((f) => {
    if (f.toLowerCase().endsWith('.pdf')) {
      const key = `${path.dirname(f)}|${path.basename(f, '.pdf')}`;
      if (mdKeys.has(key)) return false;
    }
    return true;
  });
}

// --- メイン ---

async function main() {
  console.log(`[DOCS] ${config.docsDir}`);
  console.log('[INIT] 埋め込みモデル構築中（初回はダウンロードあり）...');
  const embedder = await createEmbedder({ model: config.embeddingModel });

  const files = await collectSourceFiles(config.docsDir);
  console.log(`[FILES] ${files.length} 件\n`);

  const records = [];
  for (const file of files) {
    const stem = path.basename(file, path.extname(file));
    const folder = path.relative(config.docsDir, path.dirname(file)).split(path.sep).join('/');
    let pages;
    try {
      pages = await extractPages(file);
    } catch (e) {
      console.log(`  [ERROR] ${path.basename(file)}: ${e.message}`);
      continue;
    }
    const isMd = file.toLowerCase().endsWith('.md');
    for (const { text, page } of pages) {
      const normalized = normalizeText(text);
      let chunks = isMd ? chunkMarkdown(normalized) : chunkText(normalized);
      chunks = chunks.filter((c) => meaningfulLen(c) >= MIN_CHUNK_CHARS);
      chunks.forEach((chunk, idx) => {
        records.push({
          source: path.basename(file),
          folder,
          page,
          chunkIdx: idx,
          text: `【${stem}】\n${chunk}`, // 係名ヘッダーを付与
        });
      });
    }
    console.log(`  [OK] ${path.basename(file)} (${pages.length} pages)`);
  }

  if (!records.length) {
    console.log('[WARN] チャンクが0件です');
    return;
  }

  console.log(`\n[EMBED] ${records.length} チャンクを埋め込み中...`);
  const texts = records.map((r) => r.text);
  const embeddings = [];
  for (let i = 0; i < texts.length; i += EMBED_BATCH) {
    const vecs = await embedder.embed(texts.slice(i, i + EMBED_BATCH));
    embeddings.push(...vecs);
    process.stdout.write(`\r  ${Math.min(i + EMBED_BATCH, texts.length)}/${texts.length}`);
  }
  process.stdout.write('\n');

  const dim = embeddings[0]?.length ?? 0;
  const chunks = records.map((r, i) => ({
    id: `${r.source.replace(/\.[^.]+$/, '')}_${r.page}_${r.chunkIdx}_${crypto.randomBytes(4).toString('hex')}`,
    source: r.source,
    folder: r.folder,
    page: r.page,
    chunkIdx: r.chunkIdx,
    text: r.text,
    embedding: embeddings[i],
  }));

  await writeFile(
    config.indexPath,
    JSON.stringify({
      model: config.embeddingModel,
      dim,
      builtAt: new Date().toISOString(),
      chunks,
    })
  );

  console.log('\n' + '='.repeat(50));
  console.log(`[DONE] ${chunks.length} チャンク -> ${config.indexPath}`);
  console.log(`  モデル: ${config.embeddingModel}  次元: ${dim}`);
  console.log('='.repeat(50));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
