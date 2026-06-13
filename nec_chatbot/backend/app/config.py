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

    # Embedding
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001")

    # Chroma
    CHROMA_TOP_K = int(os.getenv("CHROMA_TOP_K", "5"))

    # Gemini
    GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "500"))
    GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))

    # ログ
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


config = Config()
