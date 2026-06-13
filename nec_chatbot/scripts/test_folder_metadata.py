"""フォルダメタデータのテスト"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.rag.embedder import Embedder
from app.rag.retriever import Retriever
from app.config import config


def test_folder_metadata():
    """フォルダ情報を含めた検索テスト"""
    embedder = Embedder(model_name=config.EMBEDDING_MODEL_NAME)
    retriever = Retriever(db_path=config.CHROMA_DB_PATH)
    retriever.get_collection()

    query = 'リーダーの仕事'
    query_embedding = embedder.embed_single(query)
    results = retriever.search(query_embedding, top_k=3)

    print('[SEARCH TEST]')
    print(f'Query: {query}')
    print()

    for i, result in enumerate(results, 1):
        metadata = result.get('metadata', {})
        source = metadata.get('source', 'Unknown')
        folder = metadata.get('folder', '')
        page = metadata.get('page', '?')
        distance = result.get('distance', 0)

        if folder and folder != '.':
            source_display = f'{folder}/{source}'
        else:
            source_display = source

        text_preview = result.get('text', '')[:80].replace('\n', ' ')
        print(f'[{i}] {source_display} (p.{page}) - score: {distance:.3f}')
        print(f'    {text_preview}...')
        print()


if __name__ == "__main__":
    test_folder_metadata()
