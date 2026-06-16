// Generator ファクトリ（差し替え可能な抽象）。
// 既定は Gemini。将来ローカルLLMや抽出型を同じ generate インターフェースで追加できる。
import { createGeminiGenerator } from './geminiGenerator.js';

export function createGenerator(config) {
  // config.generator により実装を切り替える余地を残す（現状は gemini のみ）
  return createGeminiGenerator(config.gemini);
}
