// テキストのベクトル化（transformers.js / ローカル ONNX）
// Python版 embedder.py 相当。多言語MiniLMを mean-pooling + 正規化で文ベクトル化する。
import { pipeline } from '@huggingface/transformers';

export async function createEmbedder({ model }) {
  // feature-extraction パイプラインを一度だけ構築して使い回す
  const extractor = await pipeline('feature-extraction', model);

  // テキスト配列 -> ベクトル配列（number[][]）
  async function embed(texts) {
    const output = await extractor(texts, { pooling: 'mean', normalize: true });
    return output.tolist();
  }

  // 単一テキスト -> ベクトル（検索クエリ用）
  async function embedSingle(text) {
    const [vec] = await embed([text]);
    return vec;
  }

  return { embed, embedSingle };
}
