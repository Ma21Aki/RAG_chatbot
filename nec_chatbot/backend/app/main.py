"""FastAPI メインアプリケーション"""
import os
from pathlib import Path
from fastapi import FastAPI
from dotenv import load_dotenv

from .rag.embedder import Embedder
from .rag.retriever import Retriever
from .rag.generator import Generator
from .rag.pipeline import RAGPipeline

# 環境変数をロード
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

app = FastAPI(title="RAG System")

# グローバル RAG パイプラインインスタンス
rag_pipeline: RAGPipeline = None


@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時に RAG パイプラインを初期化"""
    global rag_pipeline

    gemini_api_key = os.getenv("GEMINI_API_KEY", "")  # 後で設定
    embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
    chroma_db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")

    embedder = Embedder(model_name=embedding_model)
    retriever = Retriever(db_path=chroma_db_path)
    retriever.get_collection()  # コレクション確認
    generator = Generator(api_key=gemini_api_key)

    rag_pipeline = RAGPipeline(embedder, retriever, generator)
    print("✓ RAG Pipeline initialized (Generator: Mock mode)")


@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {"status": "ok"}


@app.post("/query")
async def query(message: str):
    """RAG クエリ（後に詳細実装）

    Args:
        message: ユーザーの質問

    Returns:
        {answer, sources}
    """
    if not rag_pipeline:
        return {"error": "Pipeline not initialized"}

    result = rag_pipeline.query(message)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
