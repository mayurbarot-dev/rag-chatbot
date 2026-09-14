# Technical Docs AI Assistant

A RAG chatbot that answers engineering questions from your own technical documentation — API references, specs, runbooks — grounded strictly in **only what those documents say**.

Built for the RAG training assignment. It implements the full pipeline end to end:

```text
Document Upload → Parsing → Chunking → Embedding Generation
        → Vector Storage → Retrieval → Response Generation
```

---

## Why this use case

A generic "ask my PDF" demo hides the hard part of RAG. Technical documentation surfaces it: engineers ask precise questions (`what's the rate limit?`, `who owns service ID 2?`) where a *plausible but wrong* answer is worse than no answer. So this build focuses as much on **refusing correctly** as on answering — see [Guardrails](#guardrails) below.

---

## Quick start (Windows, one click)

Double-click **`START_PROJECT.bat`**.

It detects Python, creates the virtual environment, installs dependencies, prompts for your Gemini API key if one isn't configured, starts the server, waits for it to become healthy, and opens the browser.

You'll need a free Gemini API key from **[Google AI Studio](https://aistudio.google.com/app/apikey)**.

### Manual setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

copy .env.example .env       # then edit .env and paste your key

.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

---

## Try it in 60 seconds

Two sample PDFs ship in [`sample_docs/`](sample_docs/) — a fictional payments API reference and a platform runbook. Upload both, then walk this script:

| # | Ask | What it demonstrates |
|---|---|---|
| 1 | `My name is Dev` | Long-term memory — the name is persisted to disk |
| 2 | `Hi` | Casual messages skip retrieval entirely |
| 3 | `What is my name?` | Answered from memory, not from any document |
| 4 | `What is the default rate limit?` | **Grounded answer** citing `orbital-api-reference.pdf` p.1 |
| 5 | `Who owns service ID 2?` | Grounded answer citing `platform-runbook.pdf` p.2 |
| 6 | `Who owns service ID 006?` | **Refuses** — similar records exist, but not that one |
| 7 | `What's a good recipe for risotto?` | Below the relevance threshold → honest fallback |

**Step 6 is the one to watch.** The runbook contains services `ID 1` through `ID 4`. A naive RAG system retrieves the closest record and confidently tells you `ID 006` is owned by whoever owns `ID 2`. This one doesn't.

---

## Interface

Single self-contained `index.html` — no build step, no framework, no CDN dependency.

- **Dark and light themes**, toggled from the sidebar and remembered in `localStorage`.
- **Responsive across phone, tablet and desktop.** Above 860px the sidebar is fixed; below that it becomes an off-canvas drawer opened from the hamburger, with a dimming scrim and Escape-to-close.
- **Clean answers** — retrieval scoring and chunk provenance stay server-side; the reply is just the answer.
- **Upload feedback** — drag-and-drop, real upload progress, per-file success/error toasts, animated knowledge-base counters.
- **Keyboard**: `Enter` sends, `Shift+Enter` newlines, `/` focuses the input.
- **Touch-aware** — 36px+ tap targets, 16px input font so iOS doesn't zoom on focus, safe-area insets for notched phones, and no autofocus on mobile so the keyboard doesn't ambush you.
- Honours `prefers-reduced-motion`.

Layout is verified by an automated pass at 360/390/820/844/1366/1920px in both themes, checking for horizontal overflow, off-screen composers, and undersized touch targets.

---

## How it works

### Pipeline

| Stage | Implementation |
|---|---|
| **Upload** | `POST /api/upload`, multipart, multiple files, drag-and-drop UI |
| **Parsing** | `pypdf` per page, so page numbers survive into chunk metadata |
| **Chunking** | 1000 chars / 150 overlap, cut at natural boundaries (newline → sentence → space) |
| **Embedding** | Gemini `gemini-embedding-001`, batched, `RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY` task types |
| **Vector storage** | JSON file (`backend/data/index.json`) — text + vector + `{file_name, page}` |
| **Retrieval** | Vectorized NumPy cosine similarity, top-K, then two guardrails |
| **Generation** | Gemini chat model with a grounded prompt, or a separate conversational prompt |

### Request flow

```text
User question
  │
  ├─ "What is my name?"            → answered from memory
  ├─ "My name is Dev"              → saved to memory
  ├─ "Hi" / "thanks"               → conversational reply, no retrieval
  ├─ nothing indexed yet           → conversational reply
  │
  ├─ embed question → cosine search → top-K
  │    ├─ no results               → conversational reply
  │    ├─ best score < threshold   → "couldn't find that" fallback
  │    ├─ exact ID absent          → "not that identifier" fallback
  │    └─ otherwise                → grounded answer from retrieved chunks
  │
  └─ both turns appended to history
```

### Guardrails

Two checks stand between retrieval and the model. They exist because **similarity search always returns something** — relevance is not the same as correctness.

**1. Relevance threshold.** If the best cosine score is below `RELEVANCE_THRESHOLD`, the retrieval isn't trusted as evidence and nothing is passed to the model as documentation context.

**2. Exact-identifier validation.** When the question names a specific identifier (`ID 006`, `ID: 006`, `ID-006`, `record 42`), the retrieved chunks are checked for that literal identifier. If it isn't there, the system refuses rather than answering from a neighbouring record.

### Memory is not evidence

Two stores, deliberately separate:

```text
Document knowledge  →  backend/data/index.json    (vectors + page metadata)
User facts/history  →  backend/data/memory.json   (name + last 12 turns)
```

A remembered name personalizes the tone of an answer. It is never injected as documentation evidence — the grounded prompt says so explicitly. Clearing documents leaves memory intact, and vice versa.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | The single-page UI (served no-cache) |
| `GET` | `/api/health` | Status + whether an API key is configured (never calls Gemini) |
| `POST` | `/api/upload` | Index one or more PDFs |
| `POST` | `/api/ask` | Ask a question → `{answer, sources, grounded}` |
| `GET` | `/api/stats` | `{documents, chunks, files}` |
| `DELETE` | `/api/knowledge-base` | Clear indexed documents only |
| `DELETE` | `/api/memory` | Clear remembered facts and history only |

Errors: `400` for bad uploads (non-PDF, oversized, empty, unreadable/scanned), `422` for questions outside 1–2000 characters, `503` when no API key is configured, `500` for unexpected model failures.

Interactive docs at <http://127.0.0.1:8000/docs>.

---

## Configuration

All settings live in `backend/.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini key (`GEMINI_API_KEY` also accepted) |
| `CHAT_MODEL` | `gemini-3.1-flash-lite` | Generation model |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `TOP_K` | `5` | Chunks retrieved per question |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `MAX_FILE_SIZE_MB` | `20` | Upload size cap |
| `RELEVANCE_THRESHOLD` | `0.60` | Minimum cosine score to trust retrieval |
| `EMBED_BATCH_SIZE` | `32` | Chunks per embedding request |

**If a model name errors for your key or region**, list what's available and swap the value:

```python
from google import genai
for m in genai.Client(api_key="...").models.list():
    print(m.name, m.supported_actions)
```

Tuning notes: raise `RELEVANCE_THRESHOLD` to make the assistant more conservative about answering, lower it if it refuses too readily. Raise `CHUNK_SIZE` for prose-heavy docs, lower it for dense reference tables.

---

## Tests

```bash
backend\.venv\Scripts\python.exe backend\tests\test_pipeline.py
```

39 assertions covering the whole pipeline — parsing and chunking against the real sample PDFs, cosine ranking, memory persistence and trimming, name extraction, and all nine branches of the routing tree including both guardrails. Gemini is stubbed, so the suite runs without an API key.

---

## Project structure

```text
rag-chatbot/
├── START_PROJECT.bat            One-click Windows launcher
├── README.md
├── IMPLEMENTATION-PLAN.md       The build plan this follows
├── RAG-implementation.md        Architecture specification
├── backend/
│   ├── requirements.txt
│   ├── .env.example
│   ├── data/                    index.json + memory.json (created at runtime)
│   ├── tests/test_pipeline.py
│   └── app/
│       ├── config.py            Settings, paths, API key resolution
│       ├── rag.py               Parsing, chunking, Store, Memory, RAG
│       └── main.py              FastAPI routes and validation
├── frontend/
│   └── index.html               Self-contained UI, no build step
└── sample_docs/
    ├── orbital-api-reference.pdf
    ├── platform-runbook.pdf
    └── make_samples.py          Regenerates the samples
```

---

## Known limits, and what production would need

The JSON vector store scans every stored vector on each query. That's fine for a demo and makes the retrieval math inspectable, but it does not scale.

| Area | This build | Production |
|---|---|---|
| Vector store | JSON + NumPy scan | pgvector, Qdrant, or a managed vector DB |
| Documents | PDF only, text layer required | Add DOCX/HTML/Markdown; OCR for scanned pages |
| Files | Kept on the app server | Object storage |
| Ingestion | Synchronous on upload | Background jobs with progress |
| Identity | Single implicit user | Per-user/tenant isolation on documents *and* memory |
| Security | Localhost only | Auth, rate limiting, MIME validation, malware scanning, HTTPS, secret manager, audit logs |
| Memory | Name + last 12 turns | Explicit identity, retention policy, deletion semantics, semantic recall |

Other things worth adding next: streaming responses, hybrid retrieval (BM25 + vector) for exact identifiers and error codes, a reranker over the top-K, and evaluation on a fixed question set to catch retrieval regressions.
