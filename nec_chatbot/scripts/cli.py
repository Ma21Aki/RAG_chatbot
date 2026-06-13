"""ターミナル CLI で RAG パイプラインを実験"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.rag.embedder import Embedder
from app.rag.retriever import Retriever
from app.rag.generator import Generator
from app.rag.pipeline import RAGPipeline
from app.config import config


def main():
    """CLI main"""
    print("=" * 60)
    print("RAG Pipeline CLI - Circle Manual Q&A")
    print("=" * 60)
    print()

    # API key check
    if not config.GEMINI_API_KEY:
        print("[WARN] GEMINI_API_KEY is not set (using production mode)")
        print()

    # Chroma DB path check
    if not Path(config.CHROMA_DB_PATH).exists():
        print("[ERROR] Chroma DB not found")
        print(f"        Run rebuild_index.py at {config.CHROMA_DB_PATH}")
        sys.exit(1)

    print(f"[DB] Chroma: {config.CHROMA_DB_PATH}")
    print(f"[MODEL] Embedding: {config.EMBEDDING_MODEL_NAME}")
    print(f"[SEARCH] Top-K: {config.CHROMA_TOP_K}")
    print()

    # RAG pipeline initialization
    print("[INIT] Initializing...")
    embedder = Embedder(model_name=config.EMBEDDING_MODEL_NAME)
    retriever = Retriever(db_path=config.CHROMA_DB_PATH)
    retriever.get_collection()
    generator = Generator(api_key=config.GEMINI_API_KEY)
    pipeline = RAGPipeline(embedder, retriever, generator)
    print("[OK] Ready")
    print()

    # CLI loop
    print("[INPUT] Enter your question (type 'exit' to quit, 'search' for search-only mode)")
    print("-" * 60)

    while True:
        try:
            query = input("\nQ: ").strip()

            if not query:
                continue

            if query.lower() == "exit":
                print("\n[EXIT] Goodbye")
                break

            if query.lower() == "search":
                # Search demo
                search_query = input("Search: ").strip()
                if search_query:
                    query_embedding = embedder.embed_single(search_query)
                    results = retriever.search(query_embedding, top_k=config.CHROMA_TOP_K)
                    print(f"\n[RESULTS] Found {len(results)} items:")
                    for i, result in enumerate(results, 1):
                        metadata = result.get('metadata', {})
                        source = metadata.get('source', 'Unknown')
                        folder = metadata.get('folder', '')
                        page = metadata.get('page', '?')
                        distance = result.get('distance', 0)
                        text = result.get('text', '')[:100] + "..."
                        if folder and folder != '.':
                            source_display = f"{folder}/{source}"
                        else:
                            source_display = source
                        print(f"\n  [{i}] {source_display} (p.{page}) - score: {distance:.3f}")
                        print(f"      {text}")
                continue

            # RAG query execution
            print(f"\n[PROCESSING] Generating answer...")
            result = pipeline.query(query, top_k=config.CHROMA_TOP_K)

            # Output
            print("\n[ANSWER]")
            print("-" * 60)
            print(result['answer'])
            print("-" * 60)

            if result['sources']:
                print("\n[SOURCES]")
                for source_info in result['sources']:
                    source = source_info.get('source', 'Unknown')
                    folder = source_info.get('folder', '')
                    page = source_info.get('page', '?')
                    if folder and folder != '.':
                        source_display = f"{folder}/{source}"
                    else:
                        source_display = source
                    print(f"  - {source_display} (p.{page})")
            else:
                print("\n[SOURCES] None")

        except KeyboardInterrupt:
            print("\n\n[EXIT] Goodbye")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
