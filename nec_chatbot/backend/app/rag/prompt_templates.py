"""プロンプトテンプレート"""
import re

SYSTEM_PROMPT = """あなたはサークルのマニュアル・書類に基づいた質問応答システムです。
以下のルールに従ってください：

1. 提供された【参考情報】の範囲内で回答してください
2. 参考情報に該当する内容がない場合は、「マニュアルに記載されていません」と答えてください
3. 不確実な情報は含めないでください
4. 回答は簡潔で分かりやすくしてください
5. 必要に応じて箇条書きで整理してください
"""


def _strip_header(text: str) -> str:
    """本文先頭の【係名】行を除く（出典はグループ見出しで一度だけ示すため）"""
    return re.sub(r"^【[^】]*】\n?", "", text, count=1)


def _join_overlap(a: str, b: str, max_overlap: int = 200) -> str:
    """a の末尾と b の先頭が重なっていればその分を除いて連結する（重複削減）

    重なりが無ければ改行で連結する。
    """
    limit = min(len(a), len(b), max_overlap)
    for k in range(limit, 0, -1):
        if a[-k:] == b[:k]:
            return a + b[k:]
    return a + "\n" + b


def format_context(search_results: list) -> str:
    """検索結果をコンテキストにフォーマット

    AIが読みやすいよう、次の整形を行う。
    ① 同一文書のチャンクをまとめ、文書内は (page, chunk_idx) 順に並べる
       （続き物が順番に読める。グループは関連度順を保持）
    ② 連続するチャンク（同一ページ・連番）を結合して重複を減らす
    ③ 出典はグループ見出しで一度だけ示し、本文先頭の【係名】は除く

    Args:
        search_results: [{text, metadata: {source, page, folder, chunk_idx, ...}}, ...]
            （関連度の高い順に並んでいる前提）

    Returns:
        フォーマット済みのコンテキスト文字列
    """
    if not search_results:
        return "（該当する情報がありません）"

    # ① 同一文書でグループ化（出現順＝関連度順を保持）
    groups = []  # [((folder, source), [results])]
    index = {}
    for result in search_results:
        meta = result.get('metadata', {})
        key = (meta.get('folder', ''), meta.get('source', 'Unknown'))
        if key not in index:
            index[key] = len(groups)
            groups.append((key, []))
        groups[index[key]][1].append(result)

    blocks = []
    for i, ((folder, source), results) in enumerate(groups, 1):
        source_display = f"{folder}/{source}" if folder and folder != '.' else source

        # 文書内はページ・chunk_idx 順に並べる
        results.sort(key=lambda r: (r['metadata'].get('page', 0), r['metadata'].get('chunk_idx', 0)))

        # ② 連続チャンク（同一ページ・連番）を結合
        segments = []  # [{page, idx_end, text}]
        for r in results:
            meta = r['metadata']
            page = meta.get('page', '?')
            idx = meta.get('chunk_idx', 0)
            body = _strip_header(r.get('text', '')).strip()  # ③ ヘッダー除去
            if segments and segments[-1]['page'] == page and idx == segments[-1]['idx_end'] + 1:
                segments[-1]['text'] = _join_overlap(segments[-1]['text'], body)
                segments[-1]['idx_end'] = idx
            else:
                segments.append({'page': page, 'idx_end': idx, 'text': body})

        lines = [f"[{i}] {source_display}"]
        for seg in segments:
            lines.append(f"(p.{seg['page']}) {seg['text']}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def format_sources(search_results: list) -> list:
    """検索結果から出典情報を抽出

    Args:
        search_results: [
            {text, metadata: {source, page, folder, ...}},
            ...
        ]

    Returns:
        [{source, page, folder}, ...] のリスト
    """
    sources = []
    seen = set()

    for result in search_results:
        metadata = result.get('metadata', {})
        source = metadata.get('source', 'Unknown')
        folder = metadata.get('folder', '')
        page = metadata.get('page', None)

        source_id = (source, folder, page)
        if source_id not in seen:
            sources.append({
                'source': source,
                'folder': folder,
                'page': page
            })
            seen.add(source_id)

    return sources
