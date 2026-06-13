"""Chroma を用いたベクトル検索"""
from typing import List, Dict, Any
import chromadb


class Retriever:
    """Chroma ベクトル DB で関連テキストを検索"""

    def __init__(self, db_path: str):
        """
        Args:
            db_path: Chroma DB のパス（例: ./data/chroma_db）
        """
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = None

    def get_collection(self, collection_name: str = "documents"):
        """コレクション取得（ない場合は作成）"""
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self.collection

    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        embeddings: List[List[float]] = None,
    ):
        """テキストをインデックスに追加

        Args:
            texts: テキストのリスト
            metadatas: メタデータのリスト（{source, page, ...}）
            ids: ドキュメント ID のリスト
            embeddings: 事前計算済みベクトル。検索側と同じモデルで計算した
                ベクトルを渡すことで、索引・検索のモデル不一致を防ぐ。
        """
        if self.collection is None:
            self.get_collection()

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """ベクトルで関連テキストを検索

        Args:
            query_embedding: 質問のベクトル
            top_k: 取得する上位 K 件

        Returns:
            [{text, metadata (source, page, ...)}] のリスト
        """
        if self.collection is None:
            self.get_collection()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        if not results['documents'] or not results['documents'][0]:
            return []

        # 結果をフォーマット
        formatted_results = []
        for i, doc in enumerate(results['documents'][0]):
            formatted_results.append({
                'text': doc,
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'distance': results['distances'][0][i] if results['distances'] else None
            })

        return formatted_results

    def delete_collection(self, collection_name: str = "documents"):
        """コレクションを削除（再インデックス時）"""
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass

    def persist(self):
        """DB を永続化"""
        # chromadb の persistent クライアントは自動的に永続化される
        pass
