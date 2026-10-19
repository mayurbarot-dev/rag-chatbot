"""Persistent hybrid index: dense vectors (npy) + BM25 + chunk metadata (jsonl).

Phase 1 kept 3072-dim vectors inside one JSON file; for the 200-page GEP that
would be ~50 MB of JSON parsed on every start. Vectors now live in a float32
matrix (one matmul per query) and BM25 is rebuilt in memory at load time.
Writes go to a temp file first and are swapped in with os.replace, so a crash
mid-ingest cannot leave a half-written index behind.
"""

import json
import math
import os
import re
import threading
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import config

INDEX_VERSION = 2

STOPWORDS = set(
    "a an and are as at be been but by can could did do does for from had has have how i if in "
    "into is it its me my of on or our over so than that the their them then there these they "
    "this those to was we were what when where which while who whom why will with would you your "
    "about according also any each much many more most other report document say says said".split()
)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*%?")


def tokenize(text: str) -> list[str]:
    out = []
    for tok in TOKEN_RE.findall(text.lower()):
        tok = tok.rstrip(".,")
        if not tok or tok in STOPWORDS:
            continue
        if tok.isalpha() and len(tok) > 4:
            # Light stemming so "forecasts"/"forecast", "libraries"/"library" meet.
            if tok.endswith("ies"):
                tok = tok[:-3] + "y"
            elif tok.endswith("s") and not tok.endswith("ss"):
                tok = tok[:-1]
        out.append(tok)
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lengths = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.lengths.mean()) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))
        self.idf = {
            t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.lengths / (self.avgdl or 1.0))
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                out[i] += idf * tf * (self.k1 + 1) / (tf + norm[i])
        return out

    def df(self, term: str) -> list[int]:
        return [i for i, _ in self.postings.get(term, [])]


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


class Store:
    def __init__(self, directory: Path | None = None):
        self.dir = Path(directory or config.INDEX_DIR)
        self.lock = threading.RLock()
        self.chunks: list[dict] = []
        self.parents: dict[str, dict] = {}
        self.docs: dict[str, dict] = {}
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self.bm25 = BM25([])
        self.namespace = ""
        self.load()

    # -- persistence ------------------------------------------------------

    @property
    def legacy_index(self) -> bool:
        """A Phase 1 index.json is present: its chunks are not usable here."""
        return config.LEGACY_INDEX_FILE.exists() and not self.chunks

    def load(self) -> None:
        with self.lock:
            meta = self.dir / "meta.json"
            if not meta.exists():
                self._reset()
                return
            try:
                info = json.loads(meta.read_text(encoding="utf-8"))
                if info.get("version") != INDEX_VERSION:
                    self._reset()
                    return
                self.namespace = info.get("namespace", "")
                self.docs = info.get("docs", {})
                self.chunks = [
                    json.loads(line)
                    for line in (self.dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.parents = json.loads((self.dir / "parents.json").read_text(encoding="utf-8"))
                self.matrix = np.load(self.dir / "vectors.npy")
                if len(self.matrix) != len(self.chunks):
                    raise ValueError("vector/chunk count mismatch")
            except (OSError, ValueError, json.JSONDecodeError):
                self._reset()
                return
            self._rebuild_bm25()

    def _reset(self) -> None:
        self.chunks, self.parents, self.docs = [], {}, {}
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self.bm25 = BM25([])

    def save(self) -> None:
        with self.lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            _atomic_write(
                self.dir / "chunks.jsonl",
                "\n".join(json.dumps(c, ensure_ascii=False) for c in self.chunks).encode("utf-8"),
            )
            _atomic_write(self.dir / "parents.json", json.dumps(self.parents, ensure_ascii=False).encode("utf-8"))
            tmp = self.dir / "vectors.tmp.npy"
            np.save(tmp, self.matrix)
            os.replace(tmp, self.dir / "vectors.npy")
            # meta.json last: it is what marks the index as complete.
            _atomic_write(
                self.dir / "meta.json",
                json.dumps({"version": INDEX_VERSION, "namespace": self.namespace, "docs": self.docs},
                           ensure_ascii=False, indent=1).encode("utf-8"),
            )

    def _rebuild_bm25(self) -> None:
        self.bm25 = BM25([tokenize(c.get("search_text") or c["text"]) for c in self.chunks])

    # -- mutation ---------------------------------------------------------

    def add_document(self, doc: dict, chunks: list[dict], vectors: np.ndarray,
                     parents: dict[str, dict], namespace: str) -> None:
        with self.lock:
            if self.chunks and self.namespace and namespace != self.namespace:
                raise ValueError(
                    "The index was built with a different embedding model "
                    f"({self.namespace}); clear the knowledge base before switching to {namespace}."
                )
            self._remove(doc_id=doc["doc_id"], file_name=doc["file_name"])
            self.namespace = namespace
            if self.matrix.size == 0:
                self.matrix = vectors.astype(np.float32)
            else:
                self.matrix = np.vstack([self.matrix, vectors.astype(np.float32)])
            self.chunks.extend(chunks)
            self.parents.update(parents)
            self.docs[doc["doc_id"]] = doc
            self._rebuild_bm25()
            self.save()

    def _remove(self, doc_id: str | None = None, file_name: str | None = None) -> int:
        drop = {d for d, info in self.docs.items() if d == doc_id or info.get("file_name") == file_name}
        if not drop:
            return 0
        keep = [i for i, c in enumerate(self.chunks) if c["doc_id"] not in drop]
        removed = len(self.chunks) - len(keep)
        self.chunks = [self.chunks[i] for i in keep]
        self.matrix = self.matrix[keep] if keep else np.zeros((0, 0), dtype=np.float32)
        self.parents = {k: v for k, v in self.parents.items() if v["doc_id"] not in drop}
        for d in drop:
            self.docs.pop(d, None)
        return removed

    def remove_document(self, doc_id: str) -> int:
        with self.lock:
            removed = self._remove(doc_id=doc_id)
            if removed or doc_id not in self.docs:
                self._rebuild_bm25()
                self.save()
            return removed

    def clear(self) -> None:
        with self.lock:
            self._reset()
            self.namespace = ""
            self.save()

    # -- queries ----------------------------------------------------------

    def mask(self, doc_ids: set[str] | None = None, chunk_filter=None) -> np.ndarray:
        m = np.ones(len(self.chunks), dtype=bool)
        if doc_ids:
            m &= np.array([c["doc_id"] in doc_ids for c in self.chunks], dtype=bool)
        if chunk_filter is not None:
            m &= np.array([bool(chunk_filter(c)) for c in self.chunks], dtype=bool)
        return m

    def dense(self, query_vec: np.ndarray, mask: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not self.chunks or not mask.any():
            return []
        scores = self.matrix @ query_vec.astype(np.float32)
        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if np.isfinite(scores[i])]

    def sparse(self, tokens: list[str], mask: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not self.chunks or not tokens:
            return []
        scores = np.where(mask, self.bm25.scores(tokens), 0.0)
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    # -- reporting --------------------------------------------------------

    def stats(self) -> dict:
        kinds = Counter(c["kind"] for c in self.chunks)
        return {"documents": len(self.docs), "chunks": len(self.chunks), "by_kind": dict(kinds)}

    def file_names(self) -> list[str]:
        return [d["file_name"] for d in self.docs.values()]

    def documents(self) -> list[dict]:
        out = []
        for doc_id, info in self.docs.items():
            kinds = Counter(c["kind"] for c in self.chunks if c["doc_id"] == doc_id)
            out.append({**info, "chunks": sum(kinds.values()), "by_kind": dict(kinds)})
        return out
