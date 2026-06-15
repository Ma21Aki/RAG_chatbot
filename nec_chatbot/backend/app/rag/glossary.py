"""用語注入（クエリ拡張）

質問に含まれる内輪語を用語辞典で引き、その定義を質問に連結して検索クエリを作る。
埋め込み側が内輪語を理解できず的外れな文書を拾う問題（メーリス等）への対策。
定義の出所は OCR 済みの用語辞典 Markdown（## 見出し＝用語、別名は ・ や 、 区切り）。
"""
import re
from pathlib import Path

_GLOSSARY_GLOB = "*用語辞典*.md"
_MIN_TERM_LEN = 2  # 1文字の語（F・係 等）は誤爆しやすいので対象外


class Glossary:
    """用語辞典から {用語: 定義} を作り、質問に定義を注入する"""

    def __init__(self, pdf_data_dir: str):
        """
        Args:
            pdf_data_dir: PDF/Markdown が置かれたディレクトリ。配下から用語辞典.md を探す
        """
        self.terms = {}  # 別名 -> 定義
        files = list(Path(pdf_data_dir).rglob(_GLOSSARY_GLOB))
        if files:
            self._load(files[0].read_text(encoding="utf-8"))

    def _load(self, text: str):
        """## 見出しごとに用語と定義を取り込む（別名は ・ 、 で分割）"""
        heading = None
        body = []

        def flush():
            if heading and body:
                definition = " ".join(body).strip()
                for alias in re.split(r"[・、]", heading):
                    alias = alias.strip()
                    if len(alias) >= _MIN_TERM_LEN and definition:
                        self.terms[alias] = definition

        for line in text.splitlines():
            if line.startswith("##"):
                flush()
                heading = line.lstrip("#").strip()
                body = []
            elif heading:
                body.append(line.strip())
        flush()

    def expand(self, query: str):
        """質問に出現した用語の定義を連結した検索クエリを返す

        Args:
            query: 元の質問文

        Returns:
            (expanded_query, matched_terms)。一致が無ければ (query, [])
        """
        seen_defs = set()
        definitions = []
        matched = []
        for alias, definition in self.terms.items():
            if alias in query and definition not in seen_defs:
                seen_defs.add(definition)
                definitions.append(definition)
                matched.append(alias)
        if not definitions:
            return query, []
        return query + " " + " ".join(definitions), matched
