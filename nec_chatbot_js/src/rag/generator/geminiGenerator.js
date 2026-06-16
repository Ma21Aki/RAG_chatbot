// Gemini による回答生成（Python版 generator.py 相当）
import { GoogleGenAI } from '@google/genai';

export function createGeminiGenerator({ apiKey, model, maxOutputTokens, temperature }) {
  const ai = new GoogleGenAI({ apiKey });

  async function generate({ system, context, query }) {
    const prompt = `${system}

【参考情報】
${context}

【質問】
${query}

【回答】
`;

    const response = await ai.models.generateContent({
      model,
      contents: prompt,
      config: {
        maxOutputTokens,
        temperature,
        // gemini-2.5系は「思考」が出力トークン枠を消費し回答が途中で切れる。
        // RAGの回答に思考は不要なので無効化。
        thinkingConfig: { thinkingBudget: 0 },
      },
    });

    const u = response.usageMetadata || {};
    return {
      text: response.text ?? '',
      usage: {
        input: u.promptTokenCount ?? 0,
        output: u.candidatesTokenCount ?? 0,
        total: u.totalTokenCount ?? 0,
      },
    };
  }

  return { generate };
}
