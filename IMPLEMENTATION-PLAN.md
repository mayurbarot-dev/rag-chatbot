# Implementation Plan — Technical Docs AI Assistant (RAG Chatbot)

**Assignment:** Build an end-to-end RAG chatbot demonstrating
`Document Upload → Parsing → Chunking → Embedding → Vector Storage → Retrieval → Response Generation`.

**Decisions taken before planning:**

| Decision | Choice |
|---|---|
| Architecture | Follow `RAG-implementation.md` exactly — FastAPI + pypdf + raw Gemini SDK + JSON vector store + NumPy cosine similarity |
| Use case | **Technical documentation assistant** — ingest API docs, specs, RFCs, runbooks; answer engineering questions with file + page citations |
| Models | Configurable via `.env` (`CHAT_MODEL`, `EMBEDDING_MODEL`), defaulting to `gemini-3.1-flash-lite` / `gemini-embedding-001` |

**Note on the two decisions interacting:** "follow the doc exactly" governs the *architecture* — every module, endpoint, file path, and algorithm below is what `RAG-implementation.md` specifies. The *technical docs* use case is applied where it does not change architecture: system prompt wording, UI copy, sample documents, and the demo script. No structural deviation.

---

## 1. Target file layout

```text
D:\rag-chatbot\
├── Practical_Generative_AI_Workshop.ipynb   (reference, unchanged)
├── RAG-implementation.md                    (spec, unchanged)
├── IMPLEMENTATION-PLAN.md                   (this file)
├── README.md                                (setup + demo script)
├── START_PROJECT.bat                         (one-click Windows launcher)
├── backend\
│   ├── requirements.txt
│   ├── .env.example
│   ├── .env                                  (gitignored; created at first run)
│   ├── .venv\                                (created by launcher)
│   ├── data\
│   │   ├── index.json                        (vector store)
│   │   └── memory.json                       (facts + history)
│   └── app\
│       ├── __init__.py
│       ├── config.py
│       ├── rag.py
│       └── main.py
├── frontend\
│   └── index.html                            (self-contained: HTML + CSS + JS)
└── sample_docs\
    └── (2–3 sample technical PDFs for the demo)
```

---

## 2. Build order

Nine phases, each independently verifiable. Later phases depend on earlier ones, so they run in order.

### Phase 1 — Scaffolding and configuration

Create the directory tree, `requirements.txt`, `.env.example`, and `config.py`.

`requirements.txt`:
```text
fastapi
uvicorn[standard]
python-multipart
pypdf
numpy
python-dotenv
google-genai
```

`config.py` responsibilities (per spec §3):
- `BACKEND_DIR = Path(__file__).resolve().parents[1]`
- Load `.env` explicitly from `BACKEND_DIR / ".env"`
- Resolve API key: `GOOGLE_API_KEY` first, fall back to `GEMINI_API_KEY`
- Expose tunables with defaults:

| Setting | Default | Purpose |
|---|---|---|
| `CHAT_MODEL` | `gemini-3.1-flash-lite` | Generation model |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `TOP_K` | `5` | Chunks retrieved per question |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `MAX_FILE_SIZE_MB` | `20` | Upload size cap |
| `RELEVANCE_THRESHOLD` | `0.60` | Minimum cosine score to trust retrieval |

- Paths: `DATA_DIR`, `INDEX_FILE`, `MEMORY_FILE`, `FRONTEND_FILE`
- Create `data/` on import if missing

**Verify:** `python -c "from app import config; print(config.CHAT_MODEL)"` prints the configured model.

---

### Phase 2 — Parsing and chunking (`rag.py`, part 1)

Pure functions, no network calls — so they are testable before an API key exists.

**`normalize(text)`** (spec §8) — strip null characters, collapse repeated intra-line whitespace, drop empty lines.

**`extract_pages(data: bytes)`** (spec §7) — `PdfReader(BytesIO(data))`, iterate `enumerate(reader.pages, start=1)`, call `page.extract_text()`, yield `(page_no, normalized_text)`, skipping pages with no extractable text.

**`chunk_text(text, size, overlap)`** (spec §9):
- Raise `ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")` when `overlap >= size`
- Walk the string in `size`-character windows; for each window, look backwards for a natural end boundary in priority order: newline → sentence end (`. ! ?`) → space
- If the best boundary lands too early in the window (< ~50% of `size`), keep the hard cut instead of producing a stunted chunk
- Advance the cursor by `end - overlap`

Page metadata is attached per chunk so citations can name the page: `{"file_name": ..., "page": ...}`.

**Verify:** feed a known multi-page PDF from `sample_docs/`, assert chunk count > 0, assert consecutive chunks share `overlap` characters, assert page numbers are ascending.

---

### Phase 3 — Vector store (`rag.py`, part 2)

**`Store`** class — the minimal persistent vector DB from spec §11–§12, §24.

- `load()` / `save()` against `backend/data/index.json`
- Item shape: `{"text": str, "vector": list[float], "metadata": {"file_name": str, "page": int}}`
- `add(items)` — append and persist
- `search(query_vector, top_k)` — stack all stored vectors into a NumPy matrix, compute
  `cosine = (q · V) / (||q|| · ||V||)` vectorized, sort descending, return the top `top_k` as
  `(item, score)` pairs
- `stats()` — `{"documents": <unique file_name count>, "chunks": <len(items)>}`
- `clear()` — reset to `[]` and persist
- Guard against zero-norm vectors (divide-by-zero) and an empty index (return `[]`)

**Verify:** insert three hand-written vectors, query with one of them, assert it ranks first with score ≈ 1.0.

---

### Phase 4 — Memory store (`rag.py`, part 3)

**`Memory`** class — spec §17–§20.

- File `backend/data/memory.json`, default `{"facts": {}, "history": []}`
- `set_fact(key, value)` / `get_fact(key)`
- `add_turn(role, content)` — append, then trim to the **most recent 12 turns**
- `recent(n=10)` — last 10 turns, for general-chat context
- `clear()` — reset to the default structure

Kept in a file separate from `index.json` on purpose (spec §20): a remembered name must never be presented as document evidence.

**Verify:** set a name, restart the process, read it back — the value survives.

---

### Phase 5 — Gemini integration and the RAG core (`rag.py`, part 4)

**`RAG`** class. Constructed lazily so `/api/health` can report key status without a Gemini call (spec §4).

- `__init__` — build the Gemini client, instantiate `Store` and `Memory`
- `embed(texts)` — `client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)`; batch chunk embeddings, single-call the query. Document chunks and questions use the *same* model so the vectors are comparable (spec §10).
- `index_pdf(file_name, data)` — extract → normalize → chunk → embed → `Store.add`; returns the chunk count. Raise a clear error when a PDF yields no text (scanned/image-only → suggest OCR).

**Guardrails** (spec §13–§14), the part that separates this from a naive RAG:

- `_below_threshold(results)` — if `best_score < RELEVANCE_THRESHOLD`, retrieval is not trusted as evidence
- `_exact_id_is_missing(question, results)` — regex-match identifier forms (`ID 006`, `ID: 006`, `ID-006`, `identifier 006`); if the requested identifier does not literally appear in any retrieved chunk, refuse rather than answer from a neighbouring record

**Routing helpers** (spec §15, §18):

- `_looks_like_pdf_question(q)` — keyword signal: `pdf, file, document, page, chapter, section, uploaded, id, identifier, record`
- `_is_name_question(q)` — `what's my name`, `what is my name`, `do you remember my name`, `who am i`
- `_extract_name(q)` — `my name is X`, `call me X`, `i am X`, `i'm X`, with a stop-list (`i am looking`, `i am trying`, …) to avoid false positives

**`_general_answer(message)`** (spec §16) — a separate, non-grounded Gemini prompt carrying the remembered name plus the last 10 turns. Instructed to answer greetings naturally, use remembered facts, never pretend document information exists, and stay concise.

**`ask(question)`** — implements the decision tree from spec §21, in this exact order:

```text
1. Direct memory question        → answer from memory.facts
2. Name-setting message          → save + acknowledge naturally
3. Casual conversation           → _general_answer()
4. Empty index                   → _general_answer()
5. Embed question → cosine search → top-K
6. No results                    → _general_answer()
7. Best score < threshold        → friendly fallback
8. Requested exact ID missing    → friendly fallback
9. Otherwise                     → grounded Gemini answer + sources
```

Every handled exchange appends both turns to memory history.

**Grounded prompt** (spec §22) — context block formatted as:

```text
TECHNICAL DOCUMENTATION CONTEXT:
[SOURCE 1] File: api-spec.pdf | Page: 12
<chunk text>

[SOURCE 2] ...

QUESTION:
<user question>
```

Instructions: answer document questions **only** from the supplied context; never invent documentation facts; cite the source number; the remembered name personalizes tone but is never evidence.

**Verify:** a question answerable from a sample doc returns the right fact with a citation; "What is the capital of France?" does not return a documentation answer.

---

### Phase 6 — FastAPI layer (`main.py`)

`app = FastAPI(title="Technical Docs AI Assistant", version="1.0.0")`, module-level `rag = None`, and a `get_rag()` helper that checks for the API key and constructs `RAG()` on first use (spec §4).

| Method | Path | Behaviour |
|---|---|---|
| `GET` | `/` | Serve `frontend/index.html` with `Cache-Control: no-store, no-cache, must-revalidate, max-age=0`, `Pragma: no-cache`, `Expires: 0` (spec §5) |
| `GET` | `/api/health` | `{"status": "ok", "api_key_configured": bool}` — no Gemini call |
| `POST` | `/api/upload` | `files: list[UploadFile] = File(...)`; per file: filename present → `.pdf` extension → size ≤ `MAX_FILE_SIZE_MB * 1024 * 1024` → index → count chunks |
| `POST` | `/api/ask` | Pydantic body, `question: str = Field(min_length=1, max_length=2000)`; returns `{answer, sources, grounded}` |
| `GET` | `/api/stats` | `{"documents": n, "chunks": m}` |
| `DELETE` | `/api/knowledge-base` | Clear `index.json` only — memory untouched (spec §25) |
| `DELETE` | `/api/memory` | Reset `memory.json` only — documents untouched (spec §25) |

Error handling (spec §26): HTTP 400 for non-PDF, oversized, unparseable, and text-free PDFs; HTTP 500 for unexpected model/service failures; HTTP 503 when no API key is configured.

Sources are serialized compactly (spec §23): `{"file_name": ..., "page": ..., "score": round(score, 3)}`.

**Verify:** `uvicorn app.main:app` starts; `/api/health` returns `ok`; `curl` upload of a sample PDF returns a chunk count; a follow-up `/api/ask` returns a cited answer.

---

### Phase 7 — Frontend (`frontend/index.html`)

One self-contained file — no build step, no separate server. Talks to relative paths (`/api/upload`, `/api/ask`, `/api/stats`, `/api/memory`, `/api/knowledge-base`) since it is served from the same origin.

Layout: a sidebar (upload dropzone, knowledge-base stats, Clear Documents, Clear Memory) beside a chat pane (message list, input, send).

Responsibilities (spec §27): file picker, drag-and-drop, multipart upload with progress and per-file success/error feedback, JSON question posting, user/assistant message rendering, source chips showing `file · page · score`, live stats, both clear actions with confirmation, Enter-to-send, and a responsive layout that stays usable on mobile.

Use-case styling only: developer-tool aesthetic, monospace for code spans in answers, copy-to-clipboard on assistant messages.

**Verify:** load in Chrome, hard-refresh, upload → ask → citations render; layout holds at 375px width.

---

### Phase 8 — Launcher and sample documents

`START_PROJECT.bat` (spec §28), for a non-technical Windows user:

```text
Detect Python → create backend\.venv if missing → install requirements
→ check backend\.env → prompt for API key if absent → start backend window
→ poll /api/health → open browser at http://127.0.0.1:8000
```

Backend command: `.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000` — bound to localhost deliberately for a local demo.

`sample_docs/` gets 2–3 small technical PDFs generated for the demo, including one structured-record document with `ID 1 / ID 2 / ID 3` rows so the exact-ID guardrail from spec §14 can be demonstrated live.

---

### Phase 9 — Documentation and verification pass

`README.md`: what it is, the pipeline diagram, prerequisites, one-click and manual setup, how to get a Gemini API key, endpoint reference, configuration table, the demo script from spec §32, and the production-improvements summary from spec §30 (pgvector/Qdrant, object storage, background jobs, per-tenant auth, rate limiting, secret manager, audit logs).

Then walk the full testing checklist from spec §31 — startup, upload, RAG, memory, and UI — and report results honestly, including anything that fails.

---

## 3. Assignment coverage

| Required stage | Where it lives |
|---|---|
| Document Upload | `POST /api/upload` + drag-and-drop UI (Phase 6, 7) |
| Parsing | `extract_pages()` via pypdf, page metadata preserved (Phase 2) |
| Chunking | `chunk_text()`, boundary-aware, 1000/150 configurable (Phase 2) |
| Embedding Generation | Gemini `embed_content`, same model for chunks and queries (Phase 5) |
| Vector Storage | `Store` → `index.json` (Phase 3) |
| Retrieval | NumPy cosine similarity, top-K, threshold + exact-ID validation (Phase 3, 5) |
| Response Generation | Grounded Gemini prompt with cited sources, general-chat fallback (Phase 5) |

Beyond the minimum, and called out in the README as such: persistent long-term memory separate from the document index, a relevance threshold and exact-identifier check that prevent confidently wrong answers, conversational routing so greetings skip retrieval entirely, per-answer source citations with page numbers and scores, and a one-click Windows launcher.

---

## 4. Risks and how they are handled

| Risk | Handling |
|---|---|
| No Gemini API key at runtime | `/api/health` reports it; launcher prompts for it; endpoints return 503 with a clear message rather than crashing |
| Model name unavailable for the key/region | `CHAT_MODEL` / `EMBEDDING_MODEL` are env-configurable; README documents how to list available models (notebook §3.3) and swap them |
| Scanned / image-only PDFs | Detected at index time; 400 response explaining that OCR would be required |
| JSON vector store does not scale | Acknowledged explicitly (spec §29); README documents the pgvector/Qdrant migration path |
| Embedding API rate limits on large PDFs | Chunks embedded in batches with a bounded batch size and a clear error surfaced to the UI on failure |
| Browser caching a stale `index.html` | No-store headers on `/` (spec §5) |

---

## 5. What I need from you before Phase 5

A Gemini API key (free from [Google AI Studio](https://aistudio.google.com/app/apikey)), placed in `backend/.env` as:

```text
GOOGLE_API_KEY=your_key_here
```

Phases 1–4 and 7–8 are buildable without it. Phases 5, 6, and 9 need a live key to verify end to end — I will build them regardless and flag anything left unverified.
