# Document Q&A Assistant

A RAG chatbot that answers questions from your own PDFs and **cites the section and page** each answer came from. It works on long structured reports, data-heavy sheets and narrative documents, and reads tables, charts and footnotes as well as body text.

Built for the RAG training assignment in two phases:

- **Phase 1:** the end-to-end pipeline (upload → parse → chunk → embed → store → retrieve → generate). See [plans/phase 1/](plans/phase%201/).
- **Phase 2:** advanced RAG for three deliberately different document shapes. Plan: [plans/phase 2/IMPLEMENTATION-PLAN.md](plans/phase%202/IMPLEMENTATION-PLAN.md). Measured results: [plans/phase 2/EVAL-RESULTS.md](plans/phase%202/EVAL-RESULTS.md). Testing guide: [plans/phase 2/HOW-TO-TEST.md](plans/phase%202/HOW-TO-TEST.md).
- **How it all works**, module by module with every rule and threshold: [TECHNICAL-GUIDE.md](TECHNICAL-GUIDE.md).

```text
Document → Parsing → Structure → Chunking → Embedding → Retrieval → Reranking → Context → LLM → Evaluation
```

---

## Quick start (Windows, one click)

Double-click **`START_PROJECT.bat`**.

It detects Python, creates the virtual environment, installs dependencies, asks for your Gemini API key if one isn't configured, starts the server, waits for it to become healthy, and opens the browser.

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

> **Embeddings run locally by default.** `EMBEDDING_PROVIDER=local` uses `BAAI/bge-base-en-v1.5` on your machine: no quota, and a ~200 MB one-time download on first upload. Gemini is still used for answers, reranking and chart transcription. Set `EMBEDDING_PROVIDER=gemini` to embed through the API instead. The free tier allows about 100 texts per minute plus a daily cap, and the 200-page GEP report alone produces ~830 chunks. Pacing and a persistent cache are built in either way.

---

## Try it

The three Phase 2 test documents are in [`plans/phase 2/`](plans/phase%202/). Upload all three and ask:

| Ask | What it demonstrates |
|---|---|
| `What is the growth outlook for South Asia in this report?` | **Section-aware retrieval.** The report says "SAR", never "South Asia", in that section. Glossary aliases and the section path in each chunk's header bridge the gap. |
| `What is the growth outlook for East Asia and Pacific in 2026?` | **No cross-region bleed.** The answer is 4.2%, not South Asia's 6.3%. |
| `According to the executive summary, what are the main risks to the global outlook?` | **Explicit section filter**, plus parent-window expansion for a summary. |
| `What is the iMac's power consumption in sleep mode at 115V?` | **Rebuilt table.** The answer is 0.57W, read from the table's column header. |
| `What were the iMac's baseline emissions before reductions?` | **Chart transcription.** 445 kg comes from a waterfall chart that has no text equivalent. |
| `Which power adapter was the iMac tested with for its ENERGY STAR figures?` | **Footnote linking.** "143W" lives in endnote 18, which is attached to the claim that cites it. |
| `What does the report say about the FIFA World Cup?` | **Refusal.** The reranker finds no evidence, so there's no answer. |
| `My name is Dev`, then `What is my name?` | Long-term memory, kept separate from document evidence. |

The Phase 1 samples in [`sample_docs/`](sample_docs/) still work, including the exact-ID guardrail (`Who owns service ID 006?` refuses).

---

## How it works

### Ingest pipeline

| Stage | Module | What it does |
|---|---|---|
| **Parse** | [parsing.py](backend/app/parsing.py) | PyMuPDF lines with font size, weight, position and superscript flags. Repairs broken "Th" ligatures (`Te global` → `The global`), NFKC normalisation, de-hyphenation and drop caps. Strips running headers and footers, but first harvests printed page numbers (`p.74`, `xvii`) and chapter numbers from them. Skips contents pages. Parses the abbreviations page into a glossary. |
| **Tables** | [tables.py](backend/app/tables.py) | Rebuilds rows and columns from coordinates. Handles stacked headers ("Percentage-point differences" over `2026f 2027f`), row groups, notes, and continuation tables. Joins label/value pairs printed far apart (`GPO staff ........ 1,737`). |
| **Footnotes** | [footnotes.py](backend/app/footnotes.py) | Superscript markers become refs (`STAR.¹⁸` → ref 18). Parses endnote sections and page-bottom notes, scoped per page so that per-chapter numbering doesn't collide. |
| **Structure** | [structure.py](backend/app/structure.py) | Section tree from the PDF outline. Each outline entry is snapped to its real heading, because outlines are often a page off. Adds detected headings, box overlays (`BOX 1.1 … (continued)`) and acronym aliases (`South Asia` ↔ `SAR`). Assigns a document profile: `narrative`, `long-structured` or `visual-dense`. |
| **Vision** | [vision.py](backend/app/vision.py) | Scores pages for visual data (drawings, images, figure captions, label density). Renders the top pages and transcribes their charts and infographics through Gemini into structured text. Results are cached by file hash. |
| **Chunk** | [chunking.py](backend/app/chunking.py) | Chunks never cross a section boundary and are packed from whole paragraphs (~900 chars) with a one-sentence overlap. Each chunk belongs to a ~3,500-char parent window. Kinds: `text`, `table`, `figure`, `visual`, `footnote`. A contextual header (`doc › section path · p.74`) is embedded with every chunk. |
| **Embed / store** | [llm.py](backend/app/llm.py), [store.py](backend/app/store.py) | Gemini (768-dim) or local embeddings, behind a content-addressed cache. Float32 `.npy` matrix, JSONL metadata, in-memory BM25, atomic writes, versioned index. |

### Answer pipeline ([retrieval.py](backend/app/retrieval.py), [rag.py](backend/app/rag.py))

```text
question
  ├─ memory / name / small talk / empty index        → answered without retrieval
  ├─ analyse: document routing (terms only one doc uses), explicit section filters
  │           (Chapter 3, Box 1.1, Table 2.10, executive summary), named-section boost,
  │           acronym expansion, summary / numeric intent
  ├─ dense top-40 ∪ BM25 top-40 (+ the same inside a named section) → RRF fusion
  ├─ diversity: drop near-duplicates, ≤3 chunks per page
  ├─ LLM rerank 0–3; nothing ≥2                       → "couldn't find it"
  ├─ exact-ID guardrail                               → refuses a neighbouring record
  ├─ context: parent windows for summaries, reading order, 12k-char budget
  └─ grounded answer → numeric check: unsupported figures → one regeneration, else flagged
```

### Memory is not evidence

```text
Document knowledge  →  backend/data/index/        (chunks, vectors, parents, doc records)
User facts/history  →  backend/data/memory.json   (name + last 12 turns)
Caches              →  backend/data/cache/        (embeddings, chart transcriptions)
```

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Single-page UI |
| `GET` | `/api/health` | Status, models, provider settings (never calls Gemini) |
| `POST` | `/api/upload` | Queue PDFs for indexing → `{jobs}`; `?wait=true` blocks until done |
| `GET` | `/api/jobs/{id}` | Ingestion progress: stage, done/total, warnings, errors |
| `GET` | `/api/documents` | Per document: pages, profile, chunk counts by kind, sections |
| `DELETE` | `/api/documents/{doc_id}` | Remove one document |
| `POST` | `/api/ask` | `{question, doc_ids?, debug?}` → `{answer, sources, grounded, grounding}` |
| `GET` | `/api/stats` | `{documents, chunks, by_kind, files}` |
| `DELETE` | `/api/knowledge-base` | Clear all documents (memory kept) |
| `DELETE` | `/api/memory` | Clear memory (documents kept) |

Each source carries `file_name, page, page_end, page_label, section_path, kind, score, excerpt`. `debug=true` adds the query plan, the candidate lists and the rerank scores.

---

## Configuration

Settings live in `backend/.env` (see [.env.example](backend/.env.example)).

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini key (`GEMINI_API_KEY` also accepted) |
| `CHAT_MODEL` / `VISION_MODEL` | `gemini-3.1-flash-lite` | Generation and chart transcription |
| `EMBEDDING_PROVIDER` | `local` | `local` or `gemini` (switching requires re-indexing) |
| `EMBEDDING_DIM` | `768` | Gemini output dimensionality |
| `EMBED_RPM` / `GEN_RPM` | `90` / `14` | Client-side pacing for the free tier |
| `CHUNK_TARGET` / `CHUNK_MAX` / `PARENT_MAX` | `900` / `1400` / `3500` | Chunk sizes (chars) |
| `VISION_MODE` / `VISION_MAX_PAGES` | `auto` / `30` | `off`, `auto` or `all`; per-document budget |
| `TOP_K` / `CANDIDATES` / `RERANK_TOP` | `8` / `40` / `20` | Retrieval breadth |
| `HYBRID`, `QUERY_SCOPING`, `CONTEXT_HEADERS`, `TABLES`, `FOOTNOTES`, `GROUNDING_CHECK` | `on` | Stage toggles (used for ablation) |
| `RERANK` | `llm` | `off` falls back to a cosine gate at `RELEVANCE_THRESHOLD` |
| `MAX_FILE_SIZE_MB` | `30` | Upload cap |

---

## Tests and evaluation

```bash
backend\.venv\Scripts\python.exe backend\tests\test_pipeline.py    # routing + full pipeline, fake model
backend\.venv\Scripts\python.exe backend\tests\test_documents.py   # parsing/tables/footnotes/structure on the 3 PDFs
```

Both suites run offline, without an API key. `test_documents.py` pins each document finding from the plan: SAR on p96, Table 2.10 Maldives 2026f = 0.7, footnote 18 linked to the ENERGY STAR claim, and so on.

```bash
cd backend
.venv\Scripts\python.exe eval\run_eval.py --label my-run                      # 34 questions, live model
.venv\Scripts\python.exe eval\run_eval.py --pipeline phase1 --label baseline  # frozen Phase 1 engine
.venv\Scripts\python.exe eval\run_eval.py --label no-rerank --set RERANK=off  # ablation
```

The eval works on its own data directory, so your app index is never touched. It reports pass rate, strict pass (right answer *and* the right page retrieved), hit@k, MRR, section precision, unsupported numbers and latency. The 34 questions in [eval/questions.json](backend/eval/questions.json) cover the 9 sample questions, held-out questions, traps and refusals. Latest result: 34/34 (the Phase 1 baseline scored 23/28 on the original 28).

---

## Project structure

```text
rag-chatbot/
├── START_PROJECT.bat            One-click Windows launcher
├── plans/
│   ├── phase 1/                 Build plan, architecture spec, workshop notebook
│   └── phase 2/                 Brief, test documents, implementation plan, eval results
├── backend/
│   ├── requirements.txt / .env.example
│   ├── app/                     config, llm, parsing, tables, footnotes, structure, vision,
│   │                            chunking, store, retrieval, grounding, jobs, rag, main
│   ├── eval/                    questions.json, run_eval.py, phase1_rag.py (frozen baseline)
│   ├── tests/                   test_pipeline.py, test_documents.py
│   └── data/                    index/, memory.json, cache/ (created at runtime)
├── frontend/index.html          Self-contained UI, no build step
└── sample_docs/                 Phase 1 sample PDFs
```

---

## Known limits

| Area | This build | Production |
|---|---|---|
| Vector store | npy matrix + in-memory BM25, single process | pgvector / Qdrant / OpenSearch hybrid |
| Documents | PDF with a text layer | OCR for scans; DOCX/HTML |
| Heuristics | Tables, headings and footnotes are tuned on relative signals and tested on 5 PDFs | A layout model (e.g. DocLayout-YOLO, Docling) for unseen layouts |
| Vision | Gemini transcription of top-scored pages, cached | Same, plus human review of high-stakes figures |
| Ingestion | One background worker thread | Durable job queue, resumable |
| Identity / security | Single user, localhost | Auth, tenancy, rate limits, audit |

PyMuPDF is AGPL-3.0. That's fine for this training project; a closed-source product would need a commercial licence, or `pdfplumber` + `pypdfium2`.
