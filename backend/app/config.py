"""Central configuration and paths for the Technical Docs AI Assistant."""

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent

# Load the .env that sits next to the backend, not whatever happens to be in cwd.
load_dotenv(BACKEND_DIR / ".env")


def _str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


# GOOGLE_API_KEY wins; GEMINI_API_KEY is accepted as an alias.
API_KEY = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()

CHAT_MODEL = _str("CHAT_MODEL", "gemini-3.1-flash-lite")
EMBEDDING_MODEL = _str("EMBEDDING_MODEL", "gemini-embedding-001")

TOP_K = _int("TOP_K", 5)
CHUNK_SIZE = _int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)
MAX_FILE_SIZE_MB = _int("MAX_FILE_SIZE_MB", 20)
RELEVANCE_THRESHOLD = _float("RELEVANCE_THRESHOLD", 0.60)

# Chunks per embedding request. Keeps large PDFs under the API's payload limits.
EMBED_BATCH_SIZE = _int("EMBED_BATCH_SIZE", 32)

DATA_DIR = BACKEND_DIR / "data"
INDEX_FILE = DATA_DIR / "index.json"
MEMORY_FILE = DATA_DIR / "memory.json"
FRONTEND_FILE = PROJECT_DIR / "frontend" / "index.html"

DATA_DIR.mkdir(parents=True, exist_ok=True)


def has_api_key() -> bool:
    return bool(API_KEY)
