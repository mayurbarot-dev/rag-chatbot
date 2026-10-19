"""Model access: embeddings (Gemini or local), generation, JSON and image calls.

Everything that talks to a model goes through here, so pacing, retries and
caching are applied in one place. The free Gemini tier allows roughly 100
embedded texts per minute and a small number of generate calls per minute;
hitting those limits mid-ingest used to abort the whole upload.
"""

import hashlib
import json
import logging
import random
import re
import sqlite3
import threading
import time
from collections import deque
from typing import Any

import numpy as np

from . import config

log = logging.getLogger("uvicorn.error")


class QuotaExhausted(RuntimeError):
    """The provider's daily quota is used up; waiting a minute will not help."""


# --------------------------------------------------------------------------
# Pacing and retries
# --------------------------------------------------------------------------


class RateLimiter:
    """Sliding one-minute window over a cost budget (requests or texts)."""

    def __init__(self, per_minute: int):
        self.per_minute = max(1, per_minute)
        self.events: deque[tuple[float, int]] = deque()
        self.lock = threading.Lock()

    def acquire(self, cost: int = 1) -> None:
        cost = min(cost, self.per_minute)
        while True:
            with self.lock:
                now = time.monotonic()
                while self.events and now - self.events[0][0] >= 60:
                    self.events.popleft()
                used = sum(c for _, c in self.events)
                if used + cost <= self.per_minute:
                    self.events.append((now, cost))
                    return
                wait = 60 - (now - self.events[0][0]) + 0.05
            time.sleep(max(wait, 0.05))


_RETRY_DELAY_RE = re.compile(r"retry in ([\d.]+)s|retryDelay'?\"?:\s*'?\"?(\d+)s", re.IGNORECASE)


def _status(exc: Exception) -> int | None:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def with_retries(fn, *, attempts: int = 6, what: str = "model call"):
    """Retry 429 and 5xx with the server's suggested delay or exponential backoff."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # the SDK raises several error types
            status = _status(exc)
            text = str(exc)
            if status == 429 and "PerDay" in text:
                raise QuotaExhausted(
                    "The Gemini API daily quota is exhausted for this key. Try again "
                    "tomorrow, use a paid key, or set EMBEDDING_PROVIDER=local."
                ) from exc
            retryable = status in (429, 500, 502, 503, 504) or "deadline" in text.lower()
            if not retryable or attempt == attempts - 1:
                raise
            match = _RETRY_DELAY_RE.search(text)
            if match:
                delay = float(match.group(1) or match.group(2)) + 1.0
            else:
                delay = min(60.0, 2.0 * 2**attempt)
            delay += random.uniform(0, 1.5)
            log.warning("%s failed with %s; retrying in %.1fs", what, status, delay)
            time.sleep(delay)
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------
# Embedding cache
# --------------------------------------------------------------------------


class EmbeddingCache:
    """Content-addressed vector cache, so re-indexing never re-embeds a text."""

    def __init__(self, path=None):
        path = path or (config.CACHE_DIR / "embeddings.sqlite")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
        self.lock = threading.Lock()

    @staticmethod
    def key(namespace: str, text: str) -> str:
        return hashlib.sha256(f"{namespace}\x00{text}".encode("utf-8")).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        with self.lock:
            for i in range(0, len(keys), 500):
                part = keys[i : i + 500]
                marks = ",".join("?" * len(part))
                for k, blob in self.conn.execute(
                    f"SELECT key, vec FROM emb WHERE key IN ({marks})", part
                ):
                    found[k] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        with self.lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)",
                [(k, v.astype(np.float32).tobytes()) for k, v in items.items()],
            )
            self.conn.commit()


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


# --------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------


class GeminiEmbedder:
    def __init__(self, client):
        from google.genai import types

        self.client = client
        self.types = types
        self.limiter = RateLimiter(config.EMBED_RPM)
        self.namespace = f"gemini|{config.EMBEDDING_MODEL}|{config.EMBEDDING_DIM}"

    def embed(self, texts: list[str], task: str) -> np.ndarray:
        out = []
        for i in range(0, len(texts), config.EMBED_BATCH_SIZE):
            batch = texts[i : i + config.EMBED_BATCH_SIZE]
            self.limiter.acquire(len(batch))

            def call(batch=batch):
                return self.client.models.embed_content(
                    model=config.EMBEDDING_MODEL,
                    contents=batch,
                    config=self.types.EmbedContentConfig(
                        task_type=task, output_dimensionality=config.EMBEDDING_DIM
                    ),
                )

            response = with_retries(call, what="embedding")
            out.extend(e.values for e in response.embeddings)
        return np.array(out, dtype=np.float32)


class LocalEmbedder:
    _model = None
    _lock = threading.Lock()

    def __init__(self):
        self.namespace = f"local|{config.LOCAL_EMBEDDING_MODEL}"

    def _get(self):
        with LocalEmbedder._lock:
            if LocalEmbedder._model is None:
                from fastembed import TextEmbedding

                LocalEmbedder._model = TextEmbedding(
                    config.LOCAL_EMBEDDING_MODEL, cache_dir=str(config.CACHE_DIR / "models")
                )
            return LocalEmbedder._model

    def embed(self, texts: list[str], task: str) -> np.ndarray:
        model = self._get()
        with LocalEmbedder._lock:
            if task == "RETRIEVAL_QUERY":
                vectors = list(model.query_embed(texts))
            else:
                vectors = list(model.passage_embed(texts, batch_size=32))
        return np.array(vectors, dtype=np.float32)


# --------------------------------------------------------------------------
# LLM facade
# --------------------------------------------------------------------------


class LLM:
    def __init__(self):
        from google import genai
        from google.genai import types

        self.types = types
        self.client = genai.Client(api_key=config.API_KEY) if config.API_KEY else None
        self.gen_limiter = RateLimiter(config.GEN_RPM)
        self.cache = EmbeddingCache()
        if config.EMBEDDING_PROVIDER == "local":
            self.embedder = LocalEmbedder()
        else:
            self.embedder = GeminiEmbedder(self.client)

    @property
    def embedding_namespace(self) -> str:
        return self.embedder.namespace

    # -- embeddings -------------------------------------------------------

    def _embed_cached(self, texts: list[str], task: str, progress=None) -> np.ndarray:
        ns = f"{self.embedder.namespace}|{task}"
        keys = [EmbeddingCache.key(ns, t) for t in texts]
        found = self.cache.get_many(list(set(keys)))
        missing = [i for i, k in enumerate(keys) if k not in found]
        done = len(texts) - len(missing)
        if progress:
            progress(done, len(texts))
        step = max(config.EMBED_BATCH_SIZE, 1)
        for start in range(0, len(missing), step):
            idx = missing[start : start + step]
            vectors = _normalize_rows(self.embedder.embed([texts[i] for i in idx], task))
            fresh = {keys[i]: v for i, v in zip(idx, vectors)}
            self.cache.put_many(fresh)
            found.update(fresh)
            done += len(idx)
            if progress:
                progress(done, len(texts))
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.vstack([found[k] for k in keys]).astype(np.float32)

    def embed_documents(self, texts: list[str], progress=None) -> np.ndarray:
        return self._embed_cached(texts, "RETRIEVAL_DOCUMENT", progress)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_cached([text], "RETRIEVAL_QUERY")[0]

    # -- generation -------------------------------------------------------

    def _config(self, temperature: float, schema=None):
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "automatic_function_calling": self.types.AutomaticFunctionCallingConfig(disable=True),
        }
        if schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = schema
        return self.types.GenerateContentConfig(**kwargs)

    def _call(self, contents, temperature: float, schema=None, model: str | None = None):
        if self.client is None:
            raise RuntimeError("No Gemini API key configured.")
        self.gen_limiter.acquire()
        return with_retries(
            lambda: self.client.models.generate_content(
                model=model or config.CHAT_MODEL,
                contents=contents,
                config=self._config(temperature, schema),
            ),
            what="generation",
        )

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        response = self._call(prompt, temperature)
        return (response.text or "").strip()

    def generate_json(self, prompt: str, schema, temperature: float = 0.0) -> dict:
        response = self._call(prompt, temperature, schema)
        return _parse_json(response.text)

    def describe_image(self, png: bytes, prompt: str, schema) -> dict:
        part = self.types.Part.from_bytes(data=png, mime_type="image/png")
        response = self._call([part, prompt], 0.0, schema, model=config.VISION_MODEL)
        return _parse_json(response.text)


def _parse_json(text: str | None) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text) if text else {}
