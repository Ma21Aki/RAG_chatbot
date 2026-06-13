"""Embedding 処理（Gemini Embedding API）"""
from typing import List
from google import genai
from google.genai import types


class Embedder:
    """テキストを Gemini Embedding API でベクトルに変換

    task_type を使い分けることで検索精度を高める：
      - 文書（索引側）: RETRIEVAL_DOCUMENT
      - 質問（検索側）: RETRIEVAL_QUERY
    """

    def __init__(self, api_key: str, model_name: str = "gemini-embedding-001"):
        """
        Args:
            api_key: Gemini API キー
            model_name: 使用する Embedding モデル名
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def embed(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """テキストのリストをベクトル化（索引側の文書向け）

        Args:
            texts: テキストリスト
            task_type: 用途（文書は RETRIEVAL_DOCUMENT）

        Returns:
            ベクトルのリスト (float の 2D リスト)
        """
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [embedding.values for embedding in result.embeddings]

    def embed_single(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
        """単一テキストをベクトル化（検索側の質問向け）

        Args:
            text: テキスト
            task_type: 用途（質問は RETRIEVAL_QUERY）

        Returns:
            ベクトル (float のリスト)
        """
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=[text],
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return result.embeddings[0].values
