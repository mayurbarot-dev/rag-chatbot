"""Frozen copy of the Phase 1 RAG engine, kept so eval can reproduce the baseline."""

import json
import re
from io import BytesIO
from typing import Any, Iterator

import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader

from app import config

# Phase 1 settings, pinned so the baseline does not drift with the new config.
PHASE1_CHUNK_SIZE = 1000
PHASE1_CHUNK_OVERLAP = 150
PHASE1_TOP_K = 5
PHASE1_RELEVANCE_THRESHOLD = 0.60

# --------------------------------------------------------------------------
# Parsing and chunking
# --------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Strip nulls, collapse intra-line whitespace, drop empty lines."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t\r\f\v]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def extract_pages(data: bytes) -> Iterator[tuple[int, str]]:
    """Yield (page_number, normalized_text) for every page that has extractable text."""
    reader = PdfReader(BytesIO(data))
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        text = normalize(raw)
        if text:
            yield page_no, text


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring natural boundaries.

    Boundary priority within the tail of each window: newline, sentence end, space.
    A boundary landing in the first half of the window is ignored -- a stunted chunk
    is worse than a hard cut.
    """
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + size, length)
        if end < length:
            window = text[start:end]
            min_keep = size // 2
            cut = -1
            for pattern in ("\n", (". ", "! ", "? "), " "):
                if isinstance(pattern, tuple):
                    found = max(window.rfind(p) for p in pattern)
                    found = found + 1 if found != -1 else -1
                else:
                    found = window.rfind(pattern)
                if found > min_keep:
                    cut = found
                    break
            if cut != -1:
                end = start + cut

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break
        start = max(end - overlap, start + 1)

    return chunks


# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------


class Store:
    """Minimal persistent vector store backed by a single JSON file."""

    def __init__(self, path=None):
        self.path = path or config.INDEX_FILE
        self.items: list[dict[str, Any]] = []
        self._matrix: np.ndarray | None = None
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.items = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.items = []
        else:
            self.items = []
        self._matrix = None

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items), encoding="utf-8")

    def add(self, items: list[dict[str, Any]]) -> None:
        self.items.extend(items)
        self._matrix = None
        self.save()

    def _vectors(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = np.array(
                [item["vector"] for item in self.items], dtype=np.float32
            )
        return self._matrix

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[dict, float]]:
        """Cosine similarity across the whole index, highest score first."""
        if not self.items:
            return []

        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        matrix = self._vectors()
        norms = np.linalg.norm(matrix, axis=1)
        # Guard zero-norm rows so they score 0 instead of producing NaN.
        safe = np.where(norms == 0, 1.0, norms)
        scores = (matrix @ q) / (safe * q_norm)
        scores = np.where(norms == 0, 0.0, scores)

        order = np.argsort(scores)[::-1][:top_k]
        return [(self.items[i], float(scores[i])) for i in order]

    def stats(self) -> dict[str, int]:
        names = {item["metadata"].get("file_name") for item in self.items}
        return {"documents": len(names), "chunks": len(self.items)}

    def file_names(self) -> list[str]:
        seen: list[str] = []
        for item in self.items:
            name = item["metadata"].get("file_name")
            if name and name not in seen:
                seen.append(name)
        return seen

    def remove_file(self, file_name: str) -> int:
        """Drop every chunk belonging to one file. Returns how many were removed."""
        before = len(self.items)
        self.items = [
            item
            for item in self.items
            if item["metadata"].get("file_name") != file_name
        ]
        removed = before - len(self.items)
        if removed:
            self._matrix = None
            self.save()
        return removed

    def clear(self) -> None:
        self.items = []
        self._matrix = None
        self.save()


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------

MAX_HISTORY_TURNS = 12
CONTEXT_TURNS = 10


class Memory:
    """Small persistent store for user facts and recent conversation turns.

    Deliberately separate from the vector index: a remembered fact must never
    become evidence for a document answer.
    """

    def __init__(self, path=None):
        self.path = path or config.MEMORY_FILE
        self.data: dict[str, Any] = {"facts": {}, "history": []}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                self.data = {
                    "facts": loaded.get("facts", {}),
                    "history": loaded.get("history", []),
                }
                return
            except (json.JSONDecodeError, OSError):
                pass
        self.data = {"facts": {}, "history": []}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def set_fact(self, key: str, value: str) -> None:
        self.data["facts"][key] = value
        self.save()

    def get_fact(self, key: str) -> str | None:
        return self.data["facts"].get(key)

    def add_turn(self, role: str, content: str) -> None:
        self.data["history"].append({"role": role, "content": content})
        self.data["history"] = self.data["history"][-MAX_HISTORY_TURNS:]
        self.save()

    def recent(self, n: int = CONTEXT_TURNS) -> list[dict[str, str]]:
        return self.data["history"][-n:]

    def clear(self) -> None:
        self.data = {"facts": {}, "history": []}
        self.save()


# --------------------------------------------------------------------------
# Routing patterns
# --------------------------------------------------------------------------

DOC_KEYWORDS = {
    "pdf", "file", "document", "documentation", "docs", "page", "chapter",
    "section", "uploaded", "id", "identifier", "record", "spec", "api",
    "endpoint", "parameter", "config", "runbook",
}

NAME_QUESTION_RE = re.compile(
    r"\b(what'?s my name|what is my name|do you remember my name|who am i)\b",
    re.IGNORECASE,
)

NAME_SET_RE = re.compile(
    r"\b(?:my name is|call me|i am|i'm)\s+([A-Za-z][A-Za-z.'-]{1,30})",
    re.IGNORECASE,
)

# Words that follow "I am"/"I'm" in ordinary sentences and are not names.
NOT_NAMES = {
    "looking", "trying", "working", "wondering", "confused", "sorry", "not",
    "using", "building", "reading", "here", "just", "an", "a", "the", "going",
    "asking", "unable", "able", "interested", "curious", "done", "new", "still",
}

GREETING_RE = re.compile(
    r"^\s*(hi|hey|hello|yo|good (morning|afternoon|evening)|how are you|"
    r"thanks|thank you|thx|bye|goodbye|ok|okay|cool|nice)\b[\s!.,?]*$",
    re.IGNORECASE,
)

ID_RE = re.compile(
    r"\b(?:id|identifier|record|ticket|issue|item)\s*[:#-]?\s*([A-Za-z]{0,4}-?\d{1,6})\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# RAG
# --------------------------------------------------------------------------

GROUNDED_PROMPT = """You are a technical documentation assistant. Answer the engineer's \
question using ONLY the documentation context below.

Rules:
- Use only the supplied context. Do not use outside knowledge about the product.
- If the context does not contain the answer, say so plainly instead of guessing.
- Cite the source number(s) you used, like [SOURCE 2].
- Keep code identifiers, endpoints, and parameter names exact.
- Be concise and direct. Use short code spans or lists where they help.
{name_line}
TECHNICAL DOCUMENTATION CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

GENERAL_PROMPT = """You are a friendly technical documentation assistant. This message is \
NOT a documentation lookup, so answer conversationally.

Rules:
- Respond naturally to greetings and small talk.
- Use remembered facts about the user when relevant.
- Never claim that something appears in the uploaded documents.
- If the user seems to want document information that is unavailable, explain politely \
that you could not find it in the uploaded documentation and suggest uploading the \
relevant file.
- Keep it concise unless more detail is requested.
{name_line}{doc_line}
RECENT CONVERSATION:
{history}

USER MESSAGE:
{message}

RESPONSE:"""


class RAG:
    def __init__(self):
        if not config.has_api_key():
            raise RuntimeError("No Gemini API key configured.")
        self.client = genai.Client(api_key=config.API_KEY)
        self.store = Store()
        self.memory = Memory()

    # -- embeddings -------------------------------------------------------

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), config.EMBED_BATCH_SIZE):
            batch = texts[i : i + config.EMBED_BATCH_SIZE]
            response = self.client.models.embed_content(
                model=config.EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            vectors.extend(list(e.values) for e in response.embeddings)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    # -- indexing ---------------------------------------------------------

    def index_pdf(self, file_name: str, data: bytes) -> int:
        """Parse, chunk, embed, and store one PDF. Returns the chunk count."""
        pending: list[dict[str, Any]] = []
        for page_no, text in extract_pages(data):
            for chunk in chunk_text(text, PHASE1_CHUNK_SIZE, PHASE1_CHUNK_OVERLAP):
                pending.append(
                    {"text": chunk, "metadata": {"file_name": file_name, "page": page_no}}
                )

        if not pending:
            raise ValueError(
                f"No readable text found in '{file_name}'. It may be a scanned or "
                "image-only PDF, which would need OCR before it can be indexed."
            )

        vectors = self.embed_documents([item["text"] for item in pending])
        for item, vector in zip(pending, vectors):
            item["vector"] = vector

        # Re-uploading a file replaces its chunks. Appending instead would leave
        # the old copies in the index, and identical duplicates then crowd every
        # distinct chunk out of the top-k window.
        self.store.remove_file(file_name)
        self.store.add(pending)
        return len(pending)

    # -- guardrails -------------------------------------------------------

    @staticmethod
    def _exact_id_is_missing(question: str, results: list[tuple[dict, float]]) -> bool:
        """True when the user asked for a specific identifier that no retrieved chunk contains.

        Without this, a question about "ID 006" happily retrieves the record for
        "ID 002" and the model answers from the wrong row.
        """
        match = ID_RE.search(question)
        if not match:
            return False

        requested = match.group(1).lower().lstrip("0") or "0"
        for item, _ in results:
            for found in ID_RE.finditer(item["text"]):
                if (found.group(1).lower().lstrip("0") or "0") == requested:
                    return False
            # Also accept a bare occurrence of the number in the chunk text.
            if re.search(rf"\b0*{re.escape(requested)}\b", item["text"], re.IGNORECASE):
                return False
        return True

    @staticmethod
    def _looks_like_pdf_question(question: str) -> bool:
        words = set(re.findall(r"[a-z]+", question.lower()))
        return bool(words & DOC_KEYWORDS)

    @staticmethod
    def _is_casual(question: str) -> bool:
        return bool(GREETING_RE.match(question.strip()))

    @staticmethod
    def _extract_name(question: str) -> str | None:
        match = NAME_SET_RE.search(question)
        if not match:
            return None
        name = match.group(1).strip(" .'-")
        if not name or name.lower() in NOT_NAMES:
            return None
        return name[:1].upper() + name[1:]

    # -- generation -------------------------------------------------------

    def _generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=config.CHAT_MODEL, contents=prompt
        )
        return (response.text or "").strip()

    def _general_answer(self, message: str) -> str:
        name = self.memory.get_fact("name")
        name_line = f"\nThe user's name is {name}. Use it naturally.\n" if name else "\n"

        files = self.store.file_names()
        doc_line = (
            f"Documents currently uploaded: {', '.join(files)}.\n"
            if files
            else "No documents have been uploaded yet.\n"
        )

        history = (
            "\n".join(f"{t['role']}: {t['content']}" for t in self.memory.recent())
            or "(no previous messages)"
        )

        return self._generate(
            GENERAL_PROMPT.format(
                name_line=name_line, doc_line=doc_line, history=history, message=message
            )
        )

    def _grounded_answer(self, question: str, results: list[tuple[dict, float]]) -> str:
        blocks = []
        for i, (item, _score) in enumerate(results, start=1):
            meta = item["metadata"]
            blocks.append(
                f"[SOURCE {i}] File: {meta.get('file_name')} | Page: {meta.get('page')}\n"
                f"{item['text']}"
            )

        name = self.memory.get_fact("name")
        name_line = (
            f"\nThe user's name is {name}. You may address them by name, but their name "
            "is NOT documentation evidence.\n"
            if name
            else "\n"
        )

        return self._generate(
            GROUNDED_PROMPT.format(
                name_line=name_line, context="\n\n".join(blocks), question=question
            )
        )

    # -- main entry point -------------------------------------------------

    def ask(self, question: str) -> dict[str, Any]:
        question = question.strip()
        self.memory.add_turn("user", question)

        result = self._route(question)

        self.memory.add_turn("assistant", result["answer"])
        return result

    def _route(self, question: str) -> dict[str, Any]:
        """The decision tree. Order matters -- memory and small talk come before retrieval."""

        # 1. Direct memory question.
        if NAME_QUESTION_RE.search(question):
            name = self.memory.get_fact("name")
            answer = (
                f"Your name is {name}."
                if name
                else "I don't know your name yet - tell me and I'll remember it."
            )
            return {"answer": answer, "sources": [], "grounded": False}

        # 2. Name-setting message.
        name = self._extract_name(question)
        if name:
            self.memory.set_fact("name", name)
            return {
                "answer": f"Nice to meet you, {name}! I'll remember that. "
                "Ask me anything about your uploaded documentation.",
                "sources": [],
                "grounded": False,
            }

        # 3. Casual conversation -- do not force this through the retriever.
        if self._is_casual(question):
            return {"answer": self._general_answer(question), "sources": [], "grounded": False}

        # 4. Nothing indexed yet.
        if not self.store.items:
            return {"answer": self._general_answer(question), "sources": [], "grounded": False}

        # 5-6. Retrieve.
        results = self.store.search(self.embed_query(question), PHASE1_TOP_K)
        if not results:
            return {"answer": self._general_answer(question), "sources": [], "grounded": False}

        best_score = results[0][1]

        # 7. Retrieval too weak to be treated as evidence.
        if best_score < PHASE1_RELEVANCE_THRESHOLD:
            return {
                "answer": (
                    "I couldn't find anything relevant to that in the uploaded "
                    "documentation. If it should be in there, try rephrasing with terms "
                    "used in the docs - or upload the file that covers it."
                ),
                "sources": [],
                "grounded": False,
            }

        # 8. A specific identifier was requested but is absent from every retrieved chunk.
        if self._exact_id_is_missing(question, results):
            return {
                "answer": (
                    "I found similar records in the documentation, but not the exact "
                    "identifier you asked about. Rather than answer from a different "
                    "record, I'd rather tell you it isn't there."
                ),
                "sources": [],
                "grounded": False,
            }

        # 9. Grounded answer.
        answer = self._grounded_answer(question, results)
        sources = [
            {
                "file_name": item["metadata"].get("file_name"),
                "page": item["metadata"].get("page"),
                "score": round(score, 3),
            }
            for item, score in results
        ]
        return {"answer": answer, "sources": sources, "grounded": True}
