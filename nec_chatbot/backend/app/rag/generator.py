"""Gemini API で回答生成"""
from typing import Dict, Any
from google import genai
from google.genai import types


class Generator:
    """Gemini API を使用して回答を生成"""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Args:
            api_key: Gemini API キー
            model_name: 使用するモデル名
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(
        self,
        system_prompt: str,
        context: str,
        query: str,
        max_tokens: int = 500,
        temperature: float = 0.3
    ) -> str:
        """プロンプトから回答を生成

        Args:
            system_prompt: システムプロンプト（ハルシネーション抑止など）
            context: 参考情報（検索結果）
            query: ユーザーの質問
            max_tokens: 最大トークン数
            temperature: 温度（0.0-1.0、低いほど決定的）

        Returns:
            生成テキスト
        """
        full_prompt = f"""{system_prompt}

【参考情報】
{context}

【質問】
{query}

【回答】
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        )

        return response.text

    def generate_with_info(
        self,
        system_prompt: str,
        context: str,
        query: str,
        max_tokens: int = 500,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """プロンプトから回答を生成（トークン情報付き）

        Args:
            system_prompt: システムプロンプト（ハルシネーション抑止など）
            context: 参考情報（検索結果）
            query: ユーザーの質問
            max_tokens: 最大トークン数
            temperature: 温度（0.0-1.0、低いほど決定的）

        Returns:
            {
                'text': 生成テキスト,
                'usage': {
                    'input_tokens': 入力トークン数,
                    'output_tokens': 出力トークン数,
                    'total_tokens': 合計トークン数
                }
            }
        """
        full_prompt = f"""{system_prompt}

【参考情報】
{context}

【質問】
{query}

【回答】
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        )

        usage = response.usage_metadata
        token_info = {
            'input_tokens': usage.prompt_token_count if usage else 0,
            'output_tokens': usage.candidates_token_count if usage else 0,
            'total_tokens': usage.total_token_count if usage else 0
        }

        return {
            'text': response.text,
            'usage': token_info
        }

    def generate_with_streaming(
        self,
        system_prompt: str,
        context: str,
        query: str,
        max_tokens: int = 500,
        temperature: float = 0.3
    ):
        """ストリーミング生成（ジェネレータで逐次返す）

        Args:
            system_prompt: システムプロンプト
            context: 参考情報
            query: ユーザーの質問
            max_tokens: 最大トークン数
            temperature: 温度

        Yields:
            テキストチャンク
        """
        full_prompt = f"""{system_prompt}

【参考情報】
{context}

【質問】
{query}

【回答】
"""

        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        ):
            if chunk.text:
                yield chunk.text
