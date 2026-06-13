"""PDF インデックス再構築スクリプト"""
import os
import sys
from pathlib import Path
import argparse
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from pypdf import PdfReader
from app.rag.embedder import Embedder
from app.rag.retriever import Retriever
from app.config import config


def extract_text_from_pdf(pdf_path: str) -> list:
    """PDF からテキストを抽出（ページごと）

    Args:
        pdf_path: PDF ファイルパス

    Returns:
        [{text, page}] のリスト
    """
    reader = PdfReader(pdf_path)
    pages = []

    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text.strip():
            pages.append({
                'text': text,
                'page': page_num
            })

    return pages


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """テキストをチャンク分割

    Args:
        text: テキスト
        chunk_size: チャンクサイズ（文字数）
        overlap: オーバーラップ（文字数）

    Returns:
        チャンクのリスト
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)  # 最小1文字は進む

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(text):
            break

        start += step

    return chunks if chunks else [text]  # テキストが小さい場合はそのまま返す


def rebuild_index(pdf_dir: str, db_path: str, model_name: str = "gemini-embedding-001"):
    """PDF ディレクトリ全体をインデックスする

    Args:
        pdf_dir: PDF ディレクトリパス
        db_path: Chroma DB パス
        model_name: Embedding モデル名
    """
    print(f"[DIR] PDF directory: {pdf_dir}")
    print(f"[DB] Chroma DB path: {db_path}")
    print()

    # Embedder & Retriever 初期化
    print("[INIT] Initializing Embedding model...")
    embedder = Embedder(api_key=config.GEMINI_API_KEY, model_name=model_name)
    retriever = Retriever(db_path=db_path)

    # 既存インデックスをリセット
    print("[RESET] Deleting existing index...")
    retriever.delete_collection()
    retriever.get_collection()

    # PDF 収集
    pdf_files = list(Path(pdf_dir).rglob("*.pdf"))
    print(f"[FILES] Found {len(pdf_files)} PDF files\n")

    if not pdf_files:
        print("[WARN] No PDF files found")
        return

    # インデックス構築
    total_chunks = 0
    total_documents = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}")

        try:
            # PDF からテキスト抽出
            # パスが正しく存在するか確認
            if not pdf_path.exists():
                print(f"  [ERROR] File not found: {pdf_path}")
                continue
            pages = extract_text_from_pdf(str(pdf_path))

            for page_info in pages:
                page_text = page_info['text']
                page_num = page_info['page']

                # チャンク分割
                chunks = chunk_text(page_text, chunk_size=500, overlap=50)

                # チャンクをインデックスに追加（バッチ処理で小分けにしてメモリ不足を回避）
                batch_size = 10
                for batch_start in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[batch_start:batch_start + batch_size]
                    texts = []
                    metadatas = []
                    ids = []

                    for chunk_idx, chunk in enumerate(batch_chunks, start=batch_start):
                        texts.append(chunk)
                        folder = str(pdf_path.parent.relative_to(Path(pdf_dir)))
                        metadatas.append({
                            'source': pdf_path.name,
                            'folder': folder,
                            'page': page_num,
                            'chunk_idx': chunk_idx
                        })
                        ids.append(f"{pdf_path.stem}_{page_num}_{chunk_idx}_{uuid4().hex[:8]}")

                    # 検索側と同じ Embedder で文書ベクトルを計算して渡す
                    # （索引・検索のモデル不一致を防ぐ）
                    embeddings = embedder.embed(texts, task_type="RETRIEVAL_DOCUMENT")
                    retriever.add_documents(texts, metadatas, ids, embeddings=embeddings)

                total_chunks += len(chunks)
                total_documents += 1

            print(f"  [OK] {len(pages)} pages processed")

        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            continue

    # 永続化
    retriever.persist()

    print()
    print("=" * 50)
    print("[DONE] Index construction completed")
    print(f"  - PDF files processed: {total_documents}")
    print(f"  - Total chunks: {total_chunks}")
    print(f"  - Chroma DB path: {db_path}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF インデックス再構築")
    parser.add_argument("--pdfs", default=config.PDF_DATA_DIR, help="PDF ディレクトリパス")
    parser.add_argument("--db", default=config.CHROMA_DB_PATH, help="Chroma DB パス")
    parser.add_argument("--model", default=config.EMBEDDING_MODEL_NAME, help="Embedding モデル名")

    args = parser.parse_args()

    rebuild_index(args.pdfs, args.db, args.model)
