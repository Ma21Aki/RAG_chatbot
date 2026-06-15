"""日本語の分かち書き（Janome + ユーザー辞書）

BM25 用にテキストを内容語のリストへ分割する。ねっしーの内輪語（総リー・IB 等）が
誤分割されないよう、user_dict.csv に登録した語を優先する。
"""
from pathlib import Path
from janome.tokenizer import Tokenizer

_USER_DICT = Path(__file__).parent / "user_dict.csv"

# 検索の手がかりになる内容語だけ残す（助詞・助動詞・記号などは捨てる）
_KEEP_POS = ("名詞", "動詞", "形容詞", "副詞")

_tokenizer = Tokenizer(str(_USER_DICT), udic_type="simpledic", udic_enc="utf8")


def tokenize(text: str) -> list:
    """テキストを内容語の原形リストに分割する

    Args:
        text: 入力テキスト

    Returns:
        トークン（原形・小文字化）のリスト
    """
    tokens = []
    for token in _tokenizer.tokenize(text):
        pos = token.part_of_speech.split(",")[0]
        if pos not in _KEEP_POS:
            continue
        # 活用語は原形に正規化（した→する 等）。原形が無い語は表層形を使う。
        base = token.base_form if token.base_form and token.base_form != "*" else token.surface
        tokens.append(base.lower())
    return tokens
