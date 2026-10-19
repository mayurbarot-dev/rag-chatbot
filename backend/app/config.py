"""Central configuration and paths for the Document Q&A assistant."""

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


def _bool(name: str, default: bool) -> bool:
    value = _str(name, "on" if default else "off").lower()
    return value in {"1", "true", "yes", "on"}


# GOOGLE_API_KEY wins; GEMINI_API_KEY is accepted as an alias.
API_KEY = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()

# -- models ----------------------------------------------------------------

CHAT_MODEL = _str("CHAT_MODEL", "gemini-3.1-flash-lite")
VISION_MODEL = _str("VISION_MODEL", CHAT_MODEL)

# "local" runs an ONNX model on this machine (fastembed); "gemini" uses the
# Gemini embedding API. Local is the default because the free Gemini tier
# (~100 texts/minute plus a daily cap) cannot reliably index the three Phase 2
# reports (~1,100 chunks), and local is the configuration the eval verified.
EMBEDDING_PROVIDER = _str("EMBEDDING_PROVIDER", "local").lower()
EMBEDDING_MODEL = _str("EMBEDDING_MODEL", "gemini-embedding-001")
LOCAL_EMBEDDING_MODEL = _str("LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
# Gemini embeddings are Matryoshka-trained, so truncating to 768 keeps most of
# the quality at a quarter of the size. Ignored for the local model.
EMBEDDING_DIM = _int("EMBEDDING_DIM", 768)

# Client-side pacing so the free tier's quotas are respected instead of hit.
EMBED_RPM = _int("EMBED_RPM", 90)  # texts per minute
GEN_RPM = _int("GEN_RPM", 14)  # generate calls per minute
EMBED_BATCH_SIZE = _int("EMBED_BATCH_SIZE", 32)

# -- chunking --------------------------------------------------------------

CHUNK_TARGET = _int("CHUNK_TARGET", 900)
CHUNK_MAX = _int("CHUNK_MAX", 1400)
PARENT_MAX = _int("PARENT_MAX", 3500)
CONTEXT_HEADERS = _bool("CONTEXT_HEADERS", True)
TABLES = _bool("TABLES", True)
FOOTNOTES = _bool("FOOTNOTES", True)

# -- vision ----------------------------------------------------------------

VISION_MODE = _str("VISION_MODE", "auto").lower()  # off | auto | all
VISION_MAX_PAGES = _int("VISION_MAX_PAGES", 30)
VISION_CONCURRENCY = _int("VISION_CONCURRENCY", 3)
VISION_DPI = _int("VISION_DPI", 150)

# -- retrieval -------------------------------------------------------------

TOP_K = _int("TOP_K", 8)  # most chunks handed to the model
CANDIDATES = _int("CANDIDATES", 40)  # dense and BM25 candidates each
HYBRID = _bool("HYBRID", True)
QUERY_SCOPING = _bool("QUERY_SCOPING", True)
RERANK = _str("RERANK", "llm").lower()  # llm | off
RERANK_TOP = _int("RERANK_TOP", 20)
CONTEXT_CHAR_BUDGET = _int("CONTEXT_CHAR_BUDGET", 12000)
# Only used when RERANK=off; the reranker's own verdict gates otherwise.
RELEVANCE_THRESHOLD = _float("RELEVANCE_THRESHOLD", 0.60)
GROUNDING_CHECK = _bool("GROUNDING_CHECK", True)

MAX_FILE_SIZE_MB = _int("MAX_FILE_SIZE_MB", 30)

# -- paths -----------------------------------------------------------------

# RAG_DATA_DIR lets the eval harness and tests work on an isolated copy.
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", "").strip() or BACKEND_DIR / "data")
INDEX_DIR = DATA_DIR / "index"
LEGACY_INDEX_FILE = DATA_DIR / "index.json"
MEMORY_FILE = DATA_DIR / "memory.json"
# Caches are keyed by content, so they are shared between the app and eval runs
# regardless of RAG_DATA_DIR: re-indexing never pays twice for the same work.
CACHE_DIR = Path(os.getenv("RAG_CACHE_DIR", "").strip() or BACKEND_DIR / "data" / "cache")
FRONTEND_FILE = PROJECT_DIR / "frontend" / "index.html"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def has_api_key() -> bool:
    return bool(API_KEY)
