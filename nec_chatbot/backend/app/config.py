"""環境変数・設定管理"""
import os
from pathlib import Path
from dotenv import load_dotenv

# nec_chatbot プロジェクトルートを基準にしたパス
# config.py は nec_chatbot/backend/app/config.py に固定されているため、相対的に3階層上
_config_dir = Path(__file__) if '__file__' in dir() else Path.cwd()
PROJECT_ROOT = Path(_config_dir).resolve().parent.parent.parent

# .env をロード
env_path = PROJECT_ROOT.parent / ".env"
load_dotenv(env_path)


class Config:
    """設定クラス"""

    # Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    # パス（絶対パス）
    PDF_DATA_DIR = str(PROJECT_ROOT / "data" / "pdfs")
    CHROMA_DB_PATH = str(PROJECT_ROOT / "data" / "chroma_db")
    SQLITE_DB_PATH = str(PROJECT_ROOT / "data" / "app.db")

    # Embedding(embeddingのモデル選択。DB作成も質問文の機械語翻訳もこれ)
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

    # Chroma(検索結果上位◯件まで検索)
    CHROMA_TOP_K = int(os.getenv("CHROMA_TOP_K","10"))    

    # Gemini(geminiAPIに課す制限。使用トークン上限、創造性(高いと確率的に低い回答も生成。0~2))
    GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "500"))
    GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))

    # ログ
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


config = Config()
