"""RAG パイプライン全体"""
from typing import Dict, Any, List
import time
from .embedder import Embedder
from .retriever import Retriever
from .generator import Generator
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever
from .prompt_templates import SYSTEM_PROMPT, format_context, format_sources


class RAGPipeline:
    """質問から回答・出典まで一連のRAG処理"""

    def __init__(self, embedder: Embedder, retriever: Retriever, generator: Generator):
        """
        Args:
            embedder: Embedding エンジン
            retriever: ベクトル検索エンジン（Chroma）
            generator: 回答生成エンジン

        検索は embedding と BM25 を RRF で統合した hybrid を用いる。
        BM25 索引は retriever のチャンクから構築するため、呼び出し前に
        コレクションがロードされている前提（get_collection 済み）。
        """
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        # 語彙検索（BM25）とベクトル検索を RRF で統合した hybrid 検索を構築
        self.hybrid = HybridRetriever(embedder, retriever, BM25Retriever(retriever))

    def query(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """ユーザー質問に対して回答と出典を返す

        Args:
            user_query: ユーザーの質問
            top_k: 検索結果の上位 K 件

        Returns:
            {
                'query': ユーザー質問,
                'answer': 生成回答,
                'sources': [{source, page}, ...],
                'search_results': 検索結果の詳細
            }
        """
        # 1. hybrid 検索（embedding + BM25 を RRF 統合）
        search_results = self.hybrid.search(user_query, top_k=top_k)

        # 2. コンテキスト作成
        context = format_context(search_results)

        # 3. 回答生成
        answer = self.generator.generate(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            query=user_query,
            max_tokens=500,
            temperature=0.3
        )

        # 5. 出典情報抽出
        sources = format_sources(search_results)

        return {
            'query': user_query,
            'answer': answer,
            'sources': sources,
            'search_results': search_results
        }

    def query_streaming(self, user_query: str, top_k: int = 5):
        """ストリーミング回答（ジェネレータで逐次返す）

        Args:
            user_query: ユーザーの質問
            top_k: 検索結果の上位 K 件

        Yields:
            回答テキストのチャンク
        """
        # 1. hybrid 検索（embedding + BM25 を RRF 統合）
        search_results = self.hybrid.search(user_query, top_k=top_k)

        # 2. コンテキスト作成
        context = format_context(search_results)

        # 3. ストリーミング回答生成
        yield from self.generator.generate_with_streaming(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            query=user_query,
            max_tokens=500,
            temperature=0.3
        )

    def query_with_info(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """詳細情報付きの質問応答（処理時間・トークン数を含む）

        Args:
            user_query: ユーザーの質問
            top_k: 検索結果の上位 K 件

        Returns:
            {
                'query': ユーザー質問,
                'answer': 生成回答,
                'sources': [{source, page}, ...],
                'search_results': 検索結果の詳細,
                'metrics': {
                    'embedding_time': エンベディング処理時間,
                    'search_time': 検索処理時間,
                    'generation_time': 生成処理時間,
                    'total_time': 全体処理時間,
                    'tokens': {
                        'input_tokens': 入力トークン数,
                        'output_tokens': 出力トークン数,
                        'total_tokens': 合計トークン数
                    },
                    'num_search_results': 検索結果数
                }
            }
        """
        total_start = time.time()

        # 1. 質問をベクトル化
        embedding_start = time.time()
        query_embedding = self.embedder.embed_single(user_query)
        embedding_time = time.time() - embedding_start

        # 2. hybrid 検索（embedding + BM25 を RRF 統合。埋め込みは上で計測済みを渡す）
        search_start = time.time()
        search_results = self.hybrid.search(user_query, top_k=top_k, query_embedding=query_embedding)
        search_time = time.time() - search_start

        # 3. コンテキスト作成
        context = format_context(search_results)

        # 4. 回答生成
        generation_start = time.time()
        result = self.generator.generate_with_info(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            query=user_query,
            max_tokens=500,
            temperature=0.3
        )
        generation_time = time.time() - generation_start

        # 5. 出典情報抽出
        sources = format_sources(search_results)

        total_time = time.time() - total_start

        return {
            'query': user_query,
            'answer': result['text'],
            'sources': sources,
            'search_results': search_results,
            'metrics': {
                'embedding_time': round(embedding_time, 3),
                'search_time': round(search_time, 3),
                'generation_time': round(generation_time, 3),
                'total_time': round(total_time, 3),
                'tokens': result['usage'],
                'num_search_results': len(search_results)
            }
        }
