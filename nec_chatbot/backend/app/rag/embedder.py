"""Embedding 処理（sentence-transformers ローカルモデル）"""
from typing import List
from sentence_transformers import SentenceTransformer


class Embedder:
    """テキストをローカルの多言語モデルでベクトルに変換

    APIキー不要・無料で動作する。日本語を含む多言語に対応した
    paraphrase-multilingual-MiniLM-L12-v2 を既定とする。
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Args:
            model_name: 使用する sentence-transformers モデル名
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """テキストのリストをベクトル化（索引側の文書向け）

        Args:
            texts: テキストリスト

        Returns:
            ベクトルのリスト (float の 2D リスト)
        """
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()

    def embed_single(self, text: str) -> List[float]:
        """単一テキストをベクトル化（検索側の質問向け）

        Args:
            text: テキスト

        Returns:
            ベクトル (float のリスト)
        """
        embedding = self.model.encode([text], convert_to_tensor=False)[0]
        return embedding.tolist()
