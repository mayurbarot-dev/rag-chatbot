"""Orchestrator: the ingest pipeline, the routing tree, memory and answering.

Ingest:  parse -> tables -> key/value merge -> footnotes -> structure -> vision
         -> chunk -> embed -> store
Answer:  memory / small talk routing -> retrieve (scope, hybrid, rerank)
         -> exact-ID guardrail -> grounded generation -> numeric check
"""

import json
import re
import threading
from datetime import datetime, timezone
from typing import Any

from . import config
from .chunking import make_chunks
from .footnotes import extract_footnotes
from .grounding import unsupported_numbers
from .llm import LLM
from .parsing import parse_pdf
from .retrieval import retrieve
from .store import Store
from .structure import build_structure
from .tables import extract_tables, merge_key_values
from .vision import select_pages, transcribe

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

NOT_FOUND = (
    "I couldn't find anything in the uploaded documents that answers that. If it "
    "should be in there, try rephrasing with the terms the document uses - or upload "
    "the file that covers it."
)

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

GROUNDED_PROMPT = """You are a document assistant. Answer the question using ONLY the \
sources below, which were retrieved from the user's uploaded documents.

Rules:
- Use only the sources. Do not add outside knowledge, even if you think you know the answer.
- Give exact figures with their units and the year, period, region or product \
configuration they apply to, as written in the sources.
- The sources may cover several regions, chapters, years or configurations. Use only \
the ones the question asks about, and say which one you used.
- If the sources disagree (for example a table and a footnote), report both and say \
that they differ.
- If a figure comes from a chart transcription, say it was read from the chart.
- If the sources do not contain the answer, say plainly that the documents do not say, \
instead of guessing.
- Cite the source number(s) you used, like [SOURCE 2].
- Be concise and direct. Use a short list when it helps.
{name_line}
SOURCES:
{context}

QUESTION:
{question}
{parts}
ANSWER:"""

GENERAL_PROMPT = """You are a friendly document assistant. This message is NOT a \
document lookup, so answer conversationally.

Rules:
- Respond naturally to greetings and small talk.
- Use remembered facts about the user when relevant.
- Never claim that something appears in the uploaded documents.
- If the user seems to want document information that is unavailable, explain politely \
that you could not find it in the uploaded documents and suggest uploading the \
relevant file.
- Keep it concise unless more detail is requested.
{name_line}{doc_line}
RECENT CONVERSATION:
{history}

USER MESSAGE:
{message}

RESPONSE:"""

RETRY_NOTE = """

IMPORTANT: a previous draft of this answer contained these figures, which do not \
appear in the sources: {numbers}. Write the answer again using only figures that \
appear in the sources."""


def _source_label(i: int, b: dict) -> str:
    path = " › ".join(b["section_path"][-3:]) or "(no section)"
    pages = b["page_label"]
    pdf = f"PDF p.{b['page']}" if b["page"] == b["page_end"] else f"PDF pp.{b['page']}-{b['page_end']}"
    kind = {"table": "table", "figure": "figure", "visual": "chart transcription",
            "footnote": "footnote"}.get(b["kind"], "text")
    return f"[SOURCE {i}] {b['file_name']} · {path} · p.{pages} ({pdf}) · {kind}"


class RAG:
    def __init__(self, llm: LLM | None = None, store: Store | None = None, memory: Memory | None = None):
        if llm is None and not config.has_api_key():
            raise RuntimeError("No Gemini API key configured.")
        self.llm = llm or LLM()
        self.store = store or Store()
        self.memory = memory or Memory()
        self.ingest_lock = threading.Lock()

    # -- indexing ---------------------------------------------------------

    def index_pdf(self, file_name: str, data: bytes, progress=None) -> int:
        """Run the full ingest pipeline for one PDF. Returns the chunk count."""

        def report(stage: str, done: int = 0, total: int = 0):
            if progress:
                progress(stage, done, total)

        with self.ingest_lock:
            report("parsing")
            doc = parse_pdf(data, file_name)
            report("layout", 0, doc.page_count)
            for page in doc.pages:
                if page.skip:
                    continue
                if config.TABLES:
                    extract_tables(page, doc.body_size)
                merge_key_values(page)
            notes = extract_footnotes(doc) if config.FOOTNOTES else {}
            structure = build_structure(doc)

            warnings: list[str] = []
            visuals: dict[int, list[str]] = {}
            vision_pages = select_pages(doc)
            if vision_pages:
                report("vision", 0, len(vision_pages))
                try:
                    visuals, warnings = transcribe(
                        self.llm, data, doc, vision_pages, lambda d, t: report("vision", d, t)
                    )
                except Exception as exc:  # vision is an enrichment, never a blocker
                    warnings.append(f"Chart transcription skipped: {exc}")

            report("chunking")
            chunks, parents = make_chunks(doc, structure, notes, visuals)
            if not chunks:
                raise ValueError(
                    f"No readable text found in '{file_name}'. It may be a scanned or "
                    "image-only PDF, which would need OCR before it can be indexed."
                )

            report("embedding", 0, len(chunks))
            vectors = self.llm.embed_documents(
                [c["search_text"] for c in chunks], lambda d, t: report("embedding", d, t)
            )
            aliases = {s.title: s.aliases for s in structure.sections if s.aliases}
            record = {
                "doc_id": doc.doc_id,
                "file_name": file_name,
                "title": doc.title,
                "sha256": doc.sha256,
                "pages": doc.page_count,
                "profile": structure.profile,
                "signals": structure.signals,
                "glossary": doc.glossary,
                "aliases": aliases,
                "sections": [
                    {"title": s.title, "page": s.page, "level": len(s.path)}
                    for s in structure.sections
                    if s.kind in ("outline", "box") or (s.kind == "heading" and len(s.path) <= 1)
                ][:80],
                "footnotes": len(notes),
                "vision_pages": sorted(visuals),
                "warnings": warnings,
                "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            report("saving")
            self.store.add_document(record, chunks, vectors, parents, self.llm.embedding_namespace)
            return len(chunks)

    # -- guardrails -------------------------------------------------------

    @staticmethod
    def _exact_id_is_missing(question: str, blocks: list[dict]) -> bool:
        """True when the user asked for a specific identifier that no retrieved chunk contains.

        Without this, a question about "ID 006" happily retrieves the record for
        "ID 002" and the model answers from the wrong row.
        """
        match = ID_RE.search(question)
        if not match:
            return False

        requested = match.group(1).lower().lstrip("0") or "0"
        for block in blocks:
            text = block["text"]
            for found in ID_RE.finditer(text):
                if (found.group(1).lower().lstrip("0") or "0") == requested:
                    return False
            # Also accept a bare occurrence of the number in the chunk text.
            if re.search(rf"\b0*{re.escape(requested)}\b", text, re.IGNORECASE):
                return False
        return True

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

    def _generate(self, prompt: str, temperature: float = 0.2) -> str:
        return self.llm.generate(prompt, temperature=temperature)

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

    def _grounded_answer(self, question: str, blocks: list[dict], parts: list[str] | None = None) -> tuple[str, dict]:
        context = "\n\n".join(f"{_source_label(i, b)}\n{b['text']}" for i, b in enumerate(blocks, start=1))

        name = self.memory.get_fact("name")
        name_line = (
            f"\nThe user's name is {name}. You may address them by name, but their name "
            "is NOT document evidence.\n"
            if name
            else "\n"
        )
        # A multi-part question becomes an explicit checklist; otherwise the
        # model tends to answer only the first part (eval question G2).
        parts_block = (
            "\nThis question has several parts. Answer each one, in order, using the sources:\n"
            + "\n".join(f"{n}. {p}" for n, p in enumerate(parts, start=1)) + "\n"
            "For each part, find the source sentence that states that exact fact using the "
            "question's own key words (e.g. 'deployed' -> 'deployment'), and answer from it. "
            "Do not substitute the date or figure of a related but different event; if you "
            "mention a related event, label it as such.\n"
            if parts else ""
        )
        prompt = GROUNDED_PROMPT.format(name_line=name_line, context=context, question=question, parts=parts_block)
        answer = self._generate(prompt)

        grounding = {"checked": config.GROUNDING_CHECK, "unsupported": [], "retried": False}
        if config.GROUNDING_CHECK:
            bad = unsupported_numbers(answer, context)
            if bad:
                grounding["retried"] = True
                retry = self._generate(prompt + RETRY_NOTE.format(numbers=", ".join(bad)))
                still = unsupported_numbers(retry, context)
                if len(still) <= len(bad):
                    answer, bad = retry, still
                grounding["unsupported"] = bad
        return answer, grounding

    # -- main entry point -------------------------------------------------

    def ask(self, question: str, doc_ids: list[str] | None = None, debug: bool = False) -> dict[str, Any]:
        question = question.strip()
        self.memory.add_turn("user", question)

        result = self._route(question, doc_ids, debug)

        self.memory.add_turn("assistant", result["answer"])
        return result

    def _route(self, question: str, doc_ids: list[str] | None, debug: bool) -> dict[str, Any]:
        """The decision tree. Order matters -- memory and small talk come before retrieval."""

        def plain(answer: str, extra: dict | None = None) -> dict[str, Any]:
            out = {"answer": answer, "sources": [], "grounded": False}
            if debug and extra:
                out["debug"] = extra
            return out

        # A message can do both ("Hi, my name is Mayur. What is my name?"), so
        # the name is saved before the memory question is answered.
        name = self._extract_name(question)
        if name:
            self.memory.set_fact("name", name)

        # 1. Direct memory question.
        if NAME_QUESTION_RE.search(question):
            known = self.memory.get_fact("name")
            return plain(
                f"Your name is {known}."
                if known
                else "I don't know your name yet - tell me and I'll remember it."
            )

        # 2. Name-setting message.
        if name:
            self.memory.set_fact("name", name)
            return plain(
                f"Nice to meet you, {name}! I'll remember that. "
                "Ask me anything about your uploaded documents."
            )

        # 3. Casual conversation -- do not force this through the retriever.
        if self._is_casual(question):
            return plain(self._general_answer(question))

        # 4. Nothing indexed yet.
        if not self.store.chunks:
            return plain(self._general_answer(question))

        # 5-7. Retrieve; the reranker (or cosine gate) decides whether anything is evidence.
        found = retrieve(self.store, self.llm, question, doc_ids)
        if found.refused or not found.blocks:
            return plain(NOT_FOUND, found.debug)

        # 8. A specific identifier was requested but is absent from every retrieved chunk.
        if self._exact_id_is_missing(question, found.blocks):
            return plain(
                "I found similar records in the documents, but not the exact "
                "identifier you asked about. Rather than answer from a different "
                "record, I'd rather tell you it isn't there.",
                found.debug,
            )

        # 9. Grounded answer.
        answer, grounding = self._grounded_answer(question, found.blocks, found.plan.parts)
        sources = [
            {
                "file_name": b["file_name"],
                "doc_id": b["doc_id"],
                "page": b["page"],
                "page_end": b["page_end"],
                "page_label": b["page_label"],
                "section_path": b["section_path"],
                "kind": b["kind"],
                "score": b["score"],
                "excerpt": b["text"][:300],
            }
            for b in found.blocks
        ]
        out = {"answer": answer, "sources": sources, "grounded": True, "grounding": grounding}
        if debug:
            out["debug"] = found.debug
        return out
