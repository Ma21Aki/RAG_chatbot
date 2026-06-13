"""プロンプトテンプレート"""

SYSTEM_PROMPT = """あなたはサークルのマニュアル・書類に基づいた質問応答システムです。
以下のルールに従ってください：

1. 提供された【参考情報】の範囲内で回答してください
2. 参考情報に該当する内容がない場合は、「マニュアルに記載されていません」と答えてください
3. 不確実な情報は含めないでください
4. 回答は簡潔で分かりやすくしてください
5. 必要に応じて箇条書きで整理してください
"""


def format_context(search_results: list) -> str:
    """検索結果をコンテキストにフォーマット

    Args:
        search_results: [
            {text, metadata: {source, page, folder, ...}},
            ...
        ]

    Returns:
        フォーマット済みのコンテキスト文字列
    """
    if not search_results:
        return "（該当する情報がありません）"

    formatted = []
    for i, result in enumerate(search_results, 1):
        metadata = result.get('metadata', {})
        source = metadata.get('source', 'Unknown')
        folder = metadata.get('folder', '')
        page = metadata.get('page', '?')
        text = result.get('text', '')

        if folder and folder != '.':
            source_display = f"{folder}/{source}"
        else:
            source_display = source

        formatted.append(f"[{i}] {source_display} (p.{page})\n{text}")

    return "\n\n".join(formatted)


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
