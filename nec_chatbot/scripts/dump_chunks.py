"""索引済みチャンクの一覧出力（目視確認用）

Chroma に入っている全チャンクを CSV に書き出す。
ヘッダー【...】を除いた本文の「非空白文字数」を付け、少ない順に並べるので、
中身が空・図だけのチャンク（壊れ候補）を上から順に見つけられる。

実行: python nec_chatbot/scripts/dump_chunks.py
出力: nec_chatbot/scripts/chunks_dump.csv （Excel で開く場合は文字化けしない utf-8-sig）
"""
import sys
import csv
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import chromadb
from app.config import config

# 壊れ候補とみなす非空白文字数のしきい値（目安。実験で調整）
BROKEN_THRESHOLD = 30

OUT = Path(__file__).parent / "chunks_dump.csv"


def meaningful_len(doc: str) -> int:
    """ヘッダー【...】を除いた本文の、空白を除いた文字数"""
    body = re.sub(r"^【.*?】\n?", "", doc, count=1)  # 先頭のヘッダーを除去
    return len(re.sub(r"\s", "", body))


def main():
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    col = client.get_collection("documents")
    data = col.get(include=["documents", "metadatas"])

    rows = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        rows.append({
            "非空白文字数": meaningful_len(doc),
            "folder": meta.get("folder", ""),
            "source": meta.get("source", ""),
            "page": meta.get("page", ""),
            "chunk_idx": meta.get("chunk_idx", ""),
            "本文": doc.replace("\n", "\\n"),  # CSV1セルに収めるため改行を可視化
        })

    rows.sort(key=lambda r: r["非空白文字数"])

    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    broken = [r for r in rows if r["非空白文字数"] < BROKEN_THRESHOLD]
    print(f"出力: {OUT}  （全 {len(rows)} チャンク）")
    print(f"非空白 {BROKEN_THRESHOLD} 文字未満の壊れ候補: {len(broken)} 件\n")
    print("--- 文字数が少ない順（上位30件）---")
    for r in rows[:30]:
        print(f'{r["非空白文字数"]:>4}文字  {r["folder"]}/{r["source"]} p{r["page"]}#{r["chunk_idx"]}')


if __name__ == "__main__":
    main()
