"""Chroma DB から存在しないファイルのベクトルを削除"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.rag.retriever import Retriever
from app.config import config


def cleanup_orphaned_vectors(pdf_dir: str, db_path: str):
    """存在しないファイルのベクトルを削除

    Args:
        pdf_dir: PDF ディレクトリパス
        db_path: Chroma DB パス
    """
    print(f"[DIR] PDF directory: {pdf_dir}")
    print(f"[DB] Chroma DB path: {db_path}")
    print()

    # Retriever 初期化
    retriever = Retriever(db_path=db_path)
    collection = retriever.get_collection()

    # Chroma DB のすべてのドキュメントを取得
    print("[GET] Fetching all documents from Chroma DB...")
    all_docs = collection.get()

    if not all_docs['ids']:
        print("[INFO] No documents in Chroma DB")
        return

    print(f"[INFO] Total documents in DB: {len(all_docs['ids'])}")
    print()

    # メタデータから source を取得して、存在確認
    pdf_dir_path = Path(pdf_dir)
    orphaned_ids = []
    existing_sources = set()

    for i, doc_id in enumerate(all_docs['ids']):
        metadata = all_docs['metadatas'][i] if all_docs['metadatas'] else {}
        source = metadata.get('source', 'Unknown')

        # source ファイルが実際に存在するか確認
        # Note: 現在は 'source' はファイル名のみなので、実際のパスを探す必要がある
        found = False
        for pdf_file in pdf_dir_path.rglob(f"*{source}*"):
            if pdf_file.is_file():
                found = True
                existing_sources.add(source)
                break

        if not found:
            orphaned_ids.append((doc_id, source))
            print(f"[ORPHANED] {source} (ID: {doc_id})")

    print()
    print(f"[SUMMARY] Found {len(orphaned_ids)} orphaned documents")

    if orphaned_ids:
        # 削除確認
        confirm = input(f"\nDelete {len(orphaned_ids)} orphaned documents? (y/n): ").strip().lower()
        if confirm == 'y':
            # ID ベースで削除
            for doc_id, source in orphaned_ids:
                collection.delete(ids=[doc_id])
                print(f"  [DELETED] {source}")

            print()
            print(f"[DONE] {len(orphaned_ids)} documents deleted")

            # 削除後の統計
            remaining = collection.get()
            print(f"[INFO] Remaining documents: {len(remaining['ids'])}")
        else:
            print("[CANCEL] Deletion cancelled")
    else:
        print("[INFO] No orphaned documents found")


if __name__ == "__main__":
    cleanup_orphaned_vectors(
        pdf_dir=config.PDF_DATA_DIR,
        db_path=config.CHROMA_DB_PATH
    )
