"""PDF インデックス再構築スクリプト"""
import os
import re
import sys
from pathlib import Path
import argparse
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from pypdf import PdfReader
from app.rag.embedder import Embedder
from app.rag.retriever import Retriever
from app.config import config


# 索引に含めるチャンクの最小文字数（空白を除く）。
# これ未満は更新履歴・空ページ等の薄いチャンクとみなして除外する。実験で調整。
MIN_CHUNK_CHARS = 30


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


def extract_pages(path: Path) -> list:
    """ファイルからページ単位のテキストを取り出す（PDF / Markdown 対応）

    OCR が必要な画像 PDF は、同じ場所に同名の .md を置くことで内容を差し替える。
    .md は全体を 1 ページ扱いにする。

    Args:
        path: PDF または Markdown のパス

    Returns:
        [{text, page}] のリスト
    """
    if path.suffix.lower() == ".md":
        text = path.read_text(encoding="utf-8")
        return [{'text': text, 'page': 1}] if text.strip() else []
    return extract_text_from_pdf(str(path))


def normalize_text(text: str) -> str:
    """抽出テキストの空白を整える

    PDF 抽出は空白・改行だらけになりがちなので、行内の連続空白を1つにまとめ、
    空行を除去する。これで薄いチャンクの判定が正確になり、埋め込みも本文に集中する。

    Args:
        text: 元テキスト

    Returns:
        整形済みテキスト
    """
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t　]+", " ", line).strip()  # 行内の連続空白を圧縮
        if line:
            lines.append(line)
    return "\n".join(lines)


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


def rebuild_index(pdf_dir: str, db_path: str, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
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
    embedder = Embedder(model_name=model_name)
    retriever = Retriever(db_path=db_path)

    # 既存インデックスをリセット
    print("[RESET] Deleting existing index...")
    retriever.delete_collection()
    retriever.get_collection()

    # 索引対象を収集（PDF と、OCR 差し替え用の Markdown）
    pdf_files = list(Path(pdf_dir).rglob("*.pdf"))
    md_files = list(Path(pdf_dir).rglob("*.md"))

    # 同じ場所に同名の .md がある PDF は、抽出失敗版なので .md に差し替える
    md_keys = {(p.parent, p.stem) for p in md_files}
    pdf_files = [p for p in pdf_files if (p.parent, p.stem) not in md_keys]
    src_files = pdf_files + md_files
    print(f"[FILES] Found {len(pdf_files)} PDF + {len(md_files)} Markdown files\n")

    if not src_files:
        print("[WARN] No source files found")
        return

    # インデックス構築
    total_chunks = 0
    total_documents = 0

    for i, src_path in enumerate(src_files, 1):
        print(f"[{i}/{len(src_files)}] {src_path.name}")

        try:
            # テキスト抽出（パスの存在を確認）
            if not src_path.exists():
                print(f"  [ERROR] File not found: {src_path}")
                continue
            pages = extract_pages(src_path)

            for page_info in pages:
                page_text = normalize_text(page_info['text'])
                page_num = page_info['page']

                # チャンク分割
                chunks = chunk_text(page_text, chunk_size=500, overlap=50)
                # 薄いチャンク（更新履歴・空ページ等）を除外
                chunks = [c for c in chunks if len(re.sub(r"\s", "", c)) >= MIN_CHUNK_CHARS]

                # チャンクをインデックスに追加（バッチ処理で小分けにしてメモリ不足を回避）
                batch_size = 10
                for batch_start in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[batch_start:batch_start + batch_size]
                    texts = []
                    metadatas = []
                    ids = []

                    for chunk_idx, chunk in enumerate(batch_chunks, start=batch_start):
                        # 係名（ファイル名）をチャンク先頭に付けてから埋め込む。
                        # 本文だけだと「どの係の仕事か」がベクトルに乗らず、
                        # 別の係の似た記述と取り違えるため。
                        section = src_path.stem
                        texts.append(f"【{section}】\n{chunk}")
                        folder = str(src_path.parent.relative_to(Path(pdf_dir)))
                        metadatas.append({
                            'source': src_path.name,
                            'folder': folder,
                            'page': page_num,
                            'chunk_idx': chunk_idx
                        })
                        ids.append(f"{src_path.stem}_{page_num}_{chunk_idx}_{uuid4().hex[:8]}")

                    # 検索側と同じ Embedder で文書ベクトルを計算して渡す
                    # （索引・検索のモデル不一致を防ぐ）
                    embeddings = embedder.embed(texts)
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
