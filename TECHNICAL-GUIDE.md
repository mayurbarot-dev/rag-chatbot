# Technical Guide — How the Document Q&A Assistant Works

This guide explains the whole system from first principles to the details: what each module does, the rules it applies, the thresholds it uses and why they exist. Everything here describes the code in [backend/app/](backend/app/) as implemented. File and function names are given so you can jump straight to the code.

**Contents**

1. [The problem RAG solves](#1-the-problem-rag-solves)
2. [System overview](#2-system-overview)
3. [Configuration and paths](#3-configuration-and-paths-configpy)
4. [Model access: embeddings, generation, pacing, caching](#4-model-access-llmpy)
5. [Ingest stage 1 — Parsing](#5-ingest-stage-1--parsing-parsingpy)
6. [Ingest stage 2 — Tables and key/value pairs](#6-ingest-stage-2--tables-and-keyvalue-pairs-tablespy)
7. [Ingest stage 3 — Footnotes](#7-ingest-stage-3--footnotes-footnotespy)
8. [Ingest stage 4 — Structure: headings, sections, boxes, profile](#8-ingest-stage-4--structure-structurepy)
9. [Ingest stage 5 — Vision: charts and infographics](#9-ingest-stage-5--vision-visionpy)
10. [Ingest stage 6 — Chunking](#10-ingest-stage-6--chunking-chunkingpy)
11. [Ingest stage 7 — Embedding and storage](#11-ingest-stage-7--embedding-and-storage-storepy)
12. [Answering stage 1 — Routing](#12-answering-stage-1--routing-ragpy)
13. [Answering stage 2 — Query analysis](#13-answering-stage-2--query-analysis-retrievalpy)
14. [Answering stage 3 — Hybrid search and fusion](#14-answering-stage-3--hybrid-search-and-fusion)
15. [Answering stage 4 — Reranking and the relevance gate](#15-answering-stage-4--reranking-and-the-relevance-gate)
16. [Answering stage 5 — Context assembly](#16-answering-stage-5--context-assembly)
17. [Answering stage 6 — Generation and the grounding check](#17-answering-stage-6--generation-and-the-grounding-check)
18. [Memory](#18-memory)
19. [Background ingestion jobs](#19-background-ingestion-jobs-jobspy)
20. [HTTP API](#20-http-api-mainpy)
21. [Frontend](#21-frontend-frontendindexhtml)
22. [Testing and evaluation](#22-testing-and-evaluation)
23. [Walkthroughs: three questions end to end](#23-walkthroughs-three-questions-end-to-end)
24. [Failure modes and how each is handled](#24-failure-modes-and-how-each-is-handled)
25. [Extending the system](#25-extending-the-system)

---

## 1. The problem RAG solves

A language model knows nothing about *your* PDFs and will happily invent plausible answers. Retrieval-Augmented Generation (RAG) fixes this in two halves:

- **Ingest (once per document).** Turn the PDF into many small, searchable pieces called *chunks*. Compute a numeric fingerprint (an *embedding vector*) for each one and store it.
- **Answer (per question).** Find the few chunks most likely to contain the answer, hand them to the model as *sources*, and instruct it to answer only from them.

Almost every hard problem is in retrieval, not generation: getting the *right* chunk in front of the model. A document's shape decides what "right" requires.

| Document | Shape | What goes wrong with naive RAG |
|---|---|---|
| GPO Annual Report (82 pp.) | Narrative prose with facts in sentences | Repeated facts and page-header noise |
| World Bank GEP (200 pp.) | Chapters, regions, boxes, tables | Wrong region or chapter retrieved; region text says "SAR", never "South Asia"; contents pages outrank real content |
| iMac Environmental Report (12 pp.) | Charts, icon call-outs, tables, endnotes | Chart values unreadable as text; footnotes detached; two product configurations blended |

This codebase is one pipeline whose stages adapt to whatever shape they detect. No stage asks which file it is looking at.

---

## 2. System overview

```text
                         ┌──────────────────────── INGEST (per PDF, background job) ───────────────────────┐
 PDF bytes ─► parsing ─► tables ─► key/value merge ─► footnotes ─► structure ─► vision ─► chunking ─► embed ─► store
                         └──────────────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────────── ANSWER (per question) ─────────────────────────────────────┐
 question ─► routing ─► query analysis ─► dense + BM25 search ─► RRF fusion ─► diversity ─► LLM rerank (gate)
          ─► exact-ID guardrail ─► context assembly ─► grounded generation ─► numeric grounding check ─► answer
                         └────────────────────────────────────────────────────────────────────────────────────┘
```

| Module | Responsibility |
|---|---|
| [config.py](backend/app/config.py) | Every setting, read from `backend/.env`, plus paths |
| [llm.py](backend/app/llm.py) | Gemini client, embedders (Gemini/local), rate limiting, retries, embedding cache |
| [parsing.py](backend/app/parsing.py) | PDF → pages of styled lines; text repair; headers/footers; page labels; front matter; glossary |
| [tables.py](backend/app/tables.py) | Table reconstruction from coordinates; key/value merging |
| [footnotes.py](backend/app/footnotes.py) | Footnote definitions; marker → note resolution |
| [structure.py](backend/app/structure.py) | Headings, section tree, boxes, aliases, document profile |
| [vision.py](backend/app/vision.py) | Choosing visual pages; rendering; chart transcription; cache |
| [chunking.py](backend/app/chunking.py) | Reading-order items → chunks, parents, headers, footnote links |
| [store.py](backend/app/store.py) | Persistent vectors + BM25 + metadata |
| [retrieval.py](backend/app/retrieval.py) | Query analysis, hybrid search, fusion, rerank, context |
| [grounding.py](backend/app/grounding.py) | Numeric faithfulness check |
| [rag.py](backend/app/rag.py) | Orchestrator: ingest pipeline, routing tree, memory, prompts |
| [jobs.py](backend/app/jobs.py) | Background ingestion queue with progress |
| [main.py](backend/app/main.py) | FastAPI routes |

---

## 3. Configuration and paths ([config.py](backend/app/config.py))

`config.py` loads `backend/.env` (always the one next to the backend, never the current directory) and exposes typed values through `_str`, `_int`, `_float` and `_bool`. A malformed value falls back to its default instead of crashing. Booleans accept `1/true/yes/on`.

Settings fall into groups:

- **Models:** `CHAT_MODEL`, `VISION_MODEL` (defaults to the chat model), `EMBEDDING_PROVIDER` (`local` by default, or `gemini`), `EMBEDDING_MODEL`, `LOCAL_EMBEDDING_MODEL`, `EMBEDDING_DIM`.
- **Pacing:** `EMBED_RPM` (texts per minute), `GEN_RPM` (generate calls per minute), `EMBED_BATCH_SIZE`.
- **Chunking:** `CHUNK_TARGET`, `CHUNK_MAX`, `PARENT_MAX`, and the toggles `CONTEXT_HEADERS`, `TABLES`, `FOOTNOTES`.
- **Vision:** `VISION_MODE` (`off`/`auto`/`all`), `VISION_MAX_PAGES`, `VISION_CONCURRENCY`, `VISION_DPI`.
- **Retrieval:** `TOP_K`, `CANDIDATES`, `HYBRID`, `QUERY_SCOPING`, `RERANK`, `RERANK_TOP`, `CONTEXT_CHAR_BUDGET`, `RELEVANCE_THRESHOLD`, `GROUNDING_CHECK`.

Paths:

| Path | Contents | Override |
|---|---|---|
| `DATA_DIR` = `backend/data` | index + memory | `RAG_DATA_DIR` |
| `INDEX_DIR` = `DATA_DIR/index` | `meta.json`, `chunks.jsonl`, `parents.json`, `vectors.npy` | — |
| `MEMORY_FILE` = `DATA_DIR/memory.json` | user facts and history | — |
| `CACHE_DIR` = `backend/data/cache` | `embeddings.sqlite`, `vision/*.json`, `models/` | `RAG_CACHE_DIR` |
| `LEGACY_INDEX_FILE` = `DATA_DIR/index.json` | Phase 1 index (only detected, never read) | — |

The cache deliberately does **not** follow `RAG_DATA_DIR`. Caches are keyed by content, so the app, tests and every eval run can share them safely, and nothing is ever paid for twice.

The toggles exist because the evaluation turns stages off one at a time (ablation) through environment variables.

---

## 4. Model access ([llm.py](backend/app/llm.py))

Every model call goes through the `LLM` class, so pacing, retries and caching live in one place.

### 4.1 Rate limiting — `RateLimiter`

A sliding one-minute window of `(timestamp, cost)` events. `acquire(cost)` drops events older than 60 s. If the remaining budget can't cover `cost`, it sleeps until the oldest event expires. For embeddings, cost = number of texts in the batch, because Gemini's free tier counts each text, not each request. For generation, cost = 1 per call.

### 4.2 Retries — `with_retries`

Up to 6 attempts:

- **HTTP 429 with "PerDay"** in the message → raise `QuotaExhausted` immediately, with an actionable message. Waiting a minute can't fix a daily quota.
- **429 / 500 / 502 / 503 / 504** or a deadline error → retryable. The wait is taken from the server's `retryDelay` / "retry in Ns" hint when present (plus 1 s). Otherwise it's exponential backoff, `2·2^attempt` capped at 60 s, plus up to 1.5 s of random jitter.
- Anything else → re-raised at once.

### 4.3 Embedders

- **`GeminiEmbedder`** calls `embed_content` in batches of `EMBED_BATCH_SIZE`, with `task_type` (`RETRIEVAL_DOCUMENT` for chunks, `RETRIEVAL_QUERY` for questions) and `output_dimensionality = EMBEDDING_DIM` (768). Gemini embeddings are Matryoshka-trained, so the first 768 of 3072 dimensions still form a good embedding. They must be re-normalised, which happens below.
- **`LocalEmbedder`** runs `BAAI/bge-base-en-v1.5` through `fastembed` (ONNX Runtime, no PyTorch). The model is loaded once per process behind a lock and downloaded to `CACHE_DIR/models` on first use. `query_embed` adds BGE's query instruction prefix; `passage_embed` doesn't. That asymmetry plays the same role as Gemini's task types.

### 4.4 Embedding cache — `EmbeddingCache`

An SQLite table `emb(key, vec)`. The key is `sha256(namespace + task + text)`, where the namespace is e.g. `local|BAAI/bge-base-en-v1.5` or `gemini|gemini-embedding-001|768`. `_embed_cached`:

1. hashes every text and fetches all known vectors in one query;
2. embeds only the misses, batch by batch, and writes each batch back immediately (a crash mid-ingest loses at most one batch);
3. L2-normalises every vector (`_normalize_rows`), so cosine similarity later is a plain dot product;
4. reports progress as `(done, total)`.

Re-uploading a document, or re-indexing after a chunking change, only embeds chunks whose text actually changed.

### 4.5 Generation

- `generate(prompt, temperature)` → text.
- `generate_json(prompt, schema)` → dict. Uses `response_mime_type="application/json"` and a Pydantic `response_schema`, then parses defensively (`_parse_json` strips stray code fences).
- `describe_image(png, prompt, schema)` → dict. Sends the image as an inline part with the prompt, using `VISION_MODEL`.

Automatic function calling is disabled on every call (the SDK otherwise warns and adds overhead).

---

## 5. Ingest stage 1 — Parsing ([parsing.py](backend/app/parsing.py))

**Goal:** turn PDF bytes into a `ParsedDoc`: a list of `Page`s, each holding `Line`s in reading order, with clean text and enough style information for later stages.

### 5.1 Why PyMuPDF

The later stages need per-span font size, weight, superscript flags and bounding boxes, drawing and image counts per page, and page rendering for vision. PyMuPDF gives all of these. pypdf (Phase 1) gives a flat string, and garbled GPO's apostrophes into `�`.

### 5.2 Data model

```text
ParsedDoc: file_name, doc_id (slug-sha6), sha256, title, pages[], body_size, body_font,
           toc[(level, title, page)], glossary{acronym: expansion}, th_repair
Page:      number (1-based), width, height, lines[], label (printed page), chapter, skip,
           drawings, image_ratio, tables[], printed
Line:      text, x0,y0,x1,y1, size, bold, font, block, refs[], role, page
           role ∈ body | heading | caption | table | footnote | box
```

### 5.3 Line extraction — `_raw_lines`

For each text line PyMuPDF reports:

- **Superscript footnote markers.** A span counts as a superscript if PyMuPDF flags it (`flags & 1`), *or* it's at most 75% of the line's main font size and raised above the baseline by over 20% of that size. If it is not the line's first span and is purely numeric (`18`, or `1,2`), it's removed from the text and its numbers go into `line.refs`. A space is inserted in its place, because the marker often stood where the word gap was ("suppliers.¹² By" → "suppliers. By").
- **Missing spaces.** Some PDFs position words without a space glyph. If the horizontal gap between consecutive spans exceeds 0.18 × font size and neither side has whitespace, a space is inserted ("(two ports)with" → "(two ports) with").
- **Size** is the character-weighted mean of non-superscript spans. **Bold** means every span is bold (flag 16, or Bold/Semibold/Heavy/Black in the font name). **Block** is PyMuPDF's text-block number; later stages use it to tell a standalone line from a line inside a paragraph.

### 5.4 Visual signals — `_visual_signals`

Per page: the number of vector drawing operations (`get_cdrawings()`, cheap) and the fraction of the page covered by images. Vision (§9) and the profile (§8.6) use both.

### 5.5 Text repair — `repair_text`, `join_lines`

1. **NFKC normalisation** turns ligature characters into letters (`ﬁ` → `fi`, `ﬂ` → `fl`).
2. Soft hyphens and NUL are removed; the replacement character `�` becomes an apostrophe.
3. **"Th" ligature repair.** Some fonts map the "Th" glyph to a bare "T". GEP has 186 instances of "Te global economy". The fix is a closed dictionary (`Te→The`, `Tis→This`, `Tese→These`, `Tey→They`, …), applied only when the document shows the fault: at least 5 matches of `\b(Te|Tis|Tese|Tey)\b` across its raw text. Normal documents are never touched, which matters because "Te Deum" and similar exist.
4. Whitespace runs collapse to one space.
5. **De-hyphenation**, at paragraph join time: `confi-` + `dence` → `confidence`, but only when a lowercase letter precedes the hyphen and the next line starts lowercase. Real hyphens survive: `low-` + `Carbon`, `2027–` + `28`.

### 5.6 Drop caps — `_merge_drop_caps`

A one-character uppercase line at least 2.5× the size of the next line, followed by a lowercase start, is glued on: `T` + `he U.S. Government…` → `The U.S. Government…` (GPO).

### 5.7 Running headers and footers — `_strip_running_lines`

Candidate lines are those in the top 10% or bottom 12% of the page. Each gets a key: lowercase letters only, so digits, spaces and punctuation vanish. That makes `2 0 1 8 A N N U A L R E P O R T | U . S . G O V…` and `CHAPTER 2 GLOBAL ECONOMIC PROSPECTS | JUNE 2026 74` comparable across pages.

- A key appearing on at least `max(3, 10% of pages)` pages is a running header or footer, and is removed. The threshold is 10%, not 20%, because odd and even pages often carry different banners (GPO alternates two).
- **Harvest before deleting.** A margin line that is a bare number (`74`) or a short roman numeral (`xvii`) is the printed page number. `CHAPTER n` in a banner sets `page.chapter`. A trailing number on a banner line is kept as a fallback page number.

**Label validation — `_assign_labels`.** Arabic page numbers keep a constant offset from the PDF index within a run (GEP: printed = PDF − 22). The most common offset is found, and a harvested number is accepted only if it matches that offset or a neighbouring page's. This rejects stray margin digits, like a "6" footnote marker at the bottom of iMac p1. Roman numerals are accepted as they are. Otherwise the PDF's own page label is used (decoding PyMuPDF's `<FEFF0063>1` escape to `c1`), and failing that the page index.

Citations therefore show what the reader sees printed on the page (`p.74`) and also the PDF page (`PDF p.96`).

### 5.8 Front matter — `_detect_skip`

A page gets `skip` set, and is never chunked, when it is:

- **blank:** under 30 characters of text;
- **toc:** its first substantial line is "Contents", "Summary of Contents", "List of figures", etc., *or* at least 4 lines (and at least 25% of lines) end in dot leaders plus a number. Unskipped, GEP's list-of-tables line "2.9 South Asia forecast summary …… 76" would out-score the real content.
- **abbreviations:** it has an "Abbreviations"/"Acronyms" line and at least 8 acronym-shaped lines. The page is parsed into the glossary instead (`_parse_glossary`). Acronyms sit in a left column and expansions to their right on the same baseline, so each acronym pairs with the right-hand line whose vertical centre is within 0.6 × font size. GEP yields 38 pairs, including `SAR → South Asia`.

### 5.9 Body metrics, title and outline

- **Body size and font:** the character-weighted mode across the document (GEP: 11pt AGaramondPro).
- **Title** — `_doc_title`. The PDF metadata title, if it's sensible: 4–120 characters, not a file name, not "Untitled". Otherwise the largest text on the first two pages (at most 3 lines), provided it is at least 2pt larger than the body. Otherwise the file name. This avoids crowning a run of body text as the title.
- **Outline** — `_clean_toc`. Drops entries that are file names, longer than 100 characters, ending in `:` or `,`, starting lowercase (split-off fragments like "and Changes in…"), date subtitles ("As of September 30, 2018"), or repeated 3+ times (banners). Levels are normalised to start at 1.

A document with fewer than 50 characters in total raises "No readable text… may need OCR".

---

## 6. Ingest stage 2 — Tables and key/value pairs ([tables.py](backend/app/tables.py))

**Why not a library?** PyMuPDF's `find_tables()` was tested on all three documents and misread them: wrong headers on GEP Table 2.10, one-row fragments on the iMac power table. These PDFs do print every cell as its own line on a shared baseline, so tables are rebuilt from coordinates.

### 6.1 Rows and cells

- **Cells** — `_cells_for`. Each line is a cell. If a line is "label followed by 2+ numbers" (`Sleep 0.56W 0.57W 0.59W`), it is split into a label cell plus number cells, with x positions interpolated by character offset. A lone `$` is dropped.
- **Rows** — `_rows`. Lines sorted by vertical centre join the current row while within 0.45 × font size of it, then sort left to right.
- **Cell classes.** `NUM_CELL` matches values (`6.3`, `-0.8`, `$874,466`, `(1,234)`, `91.1%`, `0.27W`, `..`, `<1%`). `YEAR_CELL` matches column headings (`2023`, `2026f`, `2025/26e`, `25Q1`). Years are deliberately **not** values.

### 6.2 Row types

| Type | Rule |
|---|---|
| Data row | ≥2 value cells, ≤2 other cells, first cell is a label |
| Header row | 2–16 cells, none longer than 40 characters, no values, and either ≥2 year cells or all cells ≤25 characters |
| Group row | exactly one short non-numeric cell ("Calendar-year basis", "Current assets") |
| Annotation row | exactly one cell starting with `(` ("(Average including economies…)") |

### 6.3 Region growth — `extract_tables`, `_grow_region`

Starting at a data row, the region absorbs following data, header, group and annotation rows while the vertical gap stays within 3.2 × body size. Trailing non-data rows are dropped, since they belong to whatever comes next. A region needs **at least 3 data rows** to count as a table. That threshold is what stops chart axis ticks and two-cell label/value lines from being mistaken for tables.

### 6.4 Building the table — `_build`

1. **Column count** = the most common number of values per data row.
2. **Walk upward from the first data row** (up to 9 × body size): skip annotation rows, collect group rows (the closest becomes the first group label), and stop at the first header row with enough cells to the right of the label column. That row's right-hand cells name the columns; any cell over the label column becomes the label-column header (`Mode`).
3. **Super-headers.** Up to 4 rows directly above the header (within 3.5 × body size), stopping at a data row or at the caption. Fragments that share the caption's baseline but sit right of it are taken as super-header pieces (GEP prints `TABLE 2.9 …` on the left and `Percentage-point` over the last two columns on the same line). Each super-header cell is then:
   - a **subtitle** if it starts with `(`, like the units line;
   - added to the **label header** if it sits over the label column;
   - a **title** if it's centred over all value columns (within 15% of their span). iMac: "Power consumption for iMac".
   - Otherwise it's a **prefix** for every column whose centre lies under it. GEP: `Percentage-point differences from January 2026 projections 2026f`.
4. **Continuation tables.** No header, but the previous table on the page has the same column count → reuse its columns and caption + "(continued)" (GEP's Statistical Annex).
5. **Body rows** become `(group, label, values[-n_cols:])`. Trailing footnote digits are stripped from labels (`Iraq 1` → `Iraq`) and their refs recorded.
6. **Caption** — `_caption`. The nearest `TABLE n.n` line within 140pt above the table, plus following lines that aren't values. Parenthesised lines become the subtitle.
7. **Notes** — `_notes`. Lines just below that start with `Note`/`Source`, plus following small-font lines.
8. Every consumed line gets `role="table"`, so it never appears again as loose text. The table is anchored at its first line in reading order, so it sits in the right place in the document flow.

### 6.5 Serialisation

- `markdown()`: caption, subtitle, a header row, `**group**` separator rows, value rows and notes. This is what the model reads.
- `row_sentences()`: one sentence per row, with column names attached to each value. For example, `TABLE 2.10 South Asia economy forecasts (Real GDP growth…) — Calendar-year basis — Maldives: 2023: 4.9, 2024: 3.5, 2025e: 5.7, 2026f: 0.7, …`. This is what gets embedded and keyword-indexed, because "Maldives 2026" must match a cell whose column header is far away in the markdown.

### 6.6 Key/value merging — `merge_key_values`

Layouts like `GPO staff .................... 1,737` or `• Production      50%` extract as two lines. They are joined into `GPO staff: 1,737`, so a value never travels without its label. The pairing rules:

- The left line is at most 80 characters, not itself a value, and lives in a *label block*: at most 3 lines, or at least 80% of its lines short and without a full stop.
- The right line contains a digit, is at most 28 characters, and lives in a *value block*: at most 3 lines, or at least 70% of its lines are values.
- The two overlap vertically by at least 50% of the smaller line height, and the right line is to the right. The gap must be at most 45% of the page width, unless both lines come from the same PDF text block (one visual row spanning the page).

The block rules are the safety: in two-column prose, left and right lines share baselines constantly, but they belong to big multi-line paragraph blocks and are never merged.

---

## 7. Ingest stage 3 — Footnotes ([footnotes.py](backend/app/footnotes.py))

Markers were already extracted in parsing (`Line.refs`). This stage finds the **definitions**:

1. **Endnote sections.** A page whose top 20% holds a line reading "Endnotes"/"Notes"/"Footnotes", at least 1.5pt above body size, marks the page as endnotes; the line becomes a heading. On those pages, lines matching `^(\d{1,3})…` start a note **only if the number is exactly last + 1**. That's why a continuation line beginning "95–99 percent, and Platinum…" isn't mistaken for note 95. Following lines continue the current note only at the same type size (±0.4pt), and only within the same page. This keeps the iMac's 6pt copyright block out of notes 16 and 17. Endnotes are keyed `e:<n>`.
2. **Page-bottom notes.** On other pages, small lines (at least 1pt below body size) in the lower 40% of the page, starting with a number that is actually cited on that page, are keyed `p<page>:<n>`. Following small lines continue them.
3. **Resolution** — `resolve(notes, ref, page)` tries `p<page>:<n>`, then `p<page+1>:<n>` (a note spilling onto the next page), then `e:<n>`. Page-scoped keys matter because GEP restarts footnote numbering per chapter; global numbering would attach chapter 3's note 1 to chapter 1's marker 1.
4. `cited_on` records every page whose markers resolve to the note.

iMac result: 19 endnotes. Note 18 (ENERGY STAR test conditions, 143W adapter) is linked to the p7 claim "uses 58 percent less energy".

---

## 8. Ingest stage 4 — Structure ([structure.py](backend/app/structure.py))

**Goal:** give every content line a `section` id, where each section has a full path such as `["Chapter 2. Regional Outlooks", "South Asia", "Outlook"]`.

### 8.1 Captions first

Lines matching `FIGURE|TABLE|CHART|MAP n.n` get `role="caption"`, and so do wrapped continuation lines of the same style in the same block. Captions are therefore never mistaken for headings.

### 8.2 Heading detection — `detect_headings`

Each page uses **its own** body style (`_page_body`: the character-weighted mode of size and font on that page, if it has at least 400 characters of body text). The document-wide mode can be misleading: GPO's narrative is 9.75pt Helvetica, but its longer financial section is 12pt Times, so a global body size would hide every narrative heading.

A line is a heading **candidate** if its text is heading-like and it stands nearly alone (its block has at most 3 body lines, or 4 for big text). It must also satisfy one of:

- **big:** size ≥ page body + 1.5pt;
- **bold:** all spans bold and size ≥ body − 0.6pt;
- **other font:** a different font family from the body, size ≥ body − 0.3pt, a block of at most 2 lines, at most 60 characters. GEP's "Recent developments" and "Outlook" are 11pt Helvetica in an 11pt Garamond body; size and weight say nothing, the family says everything.

**Heading-like text** means: 3–100 characters, at least 3 letters making up at least half the characters, no leading bullet, dash or parenthesis, not a caption or box marker, and no trailing `.` `,` `;` `:`.

**Wrapped headings.** Consecutive candidates with the same size and weight, the same left edge (±6pt) and a small vertical gap (−0.5 to 0.8 × size; negative because large type often overlaps) are merged. "Taking responsibility for" + "our products at every stage" becomes one heading, and so do "Carbon" + "Footprint". The **merged text is then validated as a whole**. A wrapped subtitle sentence ending in a full stop is rejected and returned to body text, and its footnote refs are preserved through the merge.

**Ranks.** Each heading has a weight = size + 0.5 if bold + 0.25 if big. Distinct weights, sorted descending, become ranks 0, 1, 2, 3+ (anything below rank 3 is capped at 3).

### 8.3 Outline preparation — `_prepare_outline`

Contents entries and box entries are removed from the outline; boxes are handled by markers, §8.5. If the outline is **flat** (all one level, like GEP's 18 entries), it is re-nested. Entries matching `Ch.|Chapter|Part|Annex|Executive Summary|…` are level 1, and the rest become level 2 under the preceding one. So "South Asia" becomes a child of "Ch. 2 Regional Outlooks".

### 8.4 Snapping outline entries to real headings — `_find_anchor`

Outline destinations are only hints: GEP's outline sends "South Asia" to p95, but the heading is at the top of p96. For each entry, pages `p, p+1, p−1, p+2` are searched. Each line, and each line joined with the next (for wrapped titles), is scored by fuzzy similarity (`SequenceMatcher`) against the title. Prefix matches of 10+ characters count as 0.85, heading lines get +0.1, and other pages get −0.03. The best match at 0.82 or above becomes the anchor; otherwise the section starts at the top of its outline page.

### 8.5 The walk — `build_structure`

Lines are processed **in reading order**, not by y position: two-column pages put a right-column heading at a smaller y than left-column text that precedes it. A stack holds the current path.

- **Outline event** (an anchor line, or the start of an unanchored entry's page): leave any box, pop headings and same-or-deeper outline levels, push the section.
- **Box marker.** A line in the top 25% of the page matching `BOX 1.1 Title (continued)`. A new box saves the current stack, keeps only the top-level outline ancestors, and pushes `Box 1.1 Title`; the same box on a later page just continues. A page *without* a box marker ends the box and restores the saved stack. So does a big heading (≥ body + 3pt) appearing on a box page; GEP p35 "Global context" resumes chapter text halfway down the page.
- **Heading:** pop headings of the same or lower rank, then push. One exception: a heading whose text is part of a current outline or box title, within 2 pages of that section's start, is a divider page repeating the chapter title ("A RISING CHALLENGE") and is ignored.
- Every other line gets the current section id.

Each `Section` stores id, title (display form: `Ch. 3 …` → `Chapter 3. …`), level, kind (`root|outline|heading|box`), path, start page and aliases.

### 8.6 Aliases and profile

- **Aliases** — `_add_aliases`. A section whose title equals a glossary expansion gets the acronym: `South Asia → SAR`, `East Asia and Pacific → EAP`. Chapters get `Chapter n` and `Ch. n`; boxes get `Box n.n`.
- **Profile** — `_profile`, shown in the UI as a badge:
  - `visual-dense`: at most 40 pages and at least 30% of non-skipped pages visual (≥40 drawings or ≥25% image area). iMac.
  - `long-structured`: at least 100 pages, or at least 3 chapter entries. GEP.
  - `narrative`: everything else. GPO.

---

## 9. Ingest stage 5 — Vision ([vision.py](backend/app/vision.py))

Some facts exist only as positions and colours. The iMac's p2 waterfall chart extracts as a jumble ("–99 kg", "346 kg", "Electricity for manufacturing" three times). Vision renders such pages and has Gemini transcribe them.

### 9.1 Which pages — `score_page`, `select_pages`

| Signal | Points |
|---|---|
| ≥40 drawing operations (vector chart) | +2 |
| ≥150 drawing operations | +1 more |
| ≥20% of page covered by images | +1.5 |
| a `FIGURE`/`CHART`/`MAP` caption | +2 |
| ≥40% of lines are short (≤25 characters), like chart labels | +1.5 |
| ≥3 short lines containing digits (data labels) | +1 |
| page has a rebuilt table and score < 4 | −2 (its numbers are already structured) |

Skipped pages score 0, and so do covers (page 1–2 with under 300 characters). In `auto` mode, pages scoring at least 3 are taken in score order, up to `VISION_MAX_PAGES` (30). `all` takes every non-skipped page; `off` takes none. Result: iMac pp.1, 2, 6, 8; 30 GEP figure pages; 6 GPO pages.

### 9.2 Transcription — `transcribe`

1. **Cache check:** `CACHE_DIR/vision/<sha16>_<page>_<model>_<PROMPT_VERSION>_<dpi>.json`.
2. Render the page as PNG at `VISION_DPI` (150).
3. Send the image with `VISION_PROMPT` and the page's extracted text as a spelling hint (it may be jumbled). The prompt asks only for charts, tables, infographics, diagrams and call-outs, each with type, title, summary and **every printed label and value**. It forbids computing or inferring numbers, says to copy footnote markers, and says to use "unreadable" when unsure.
4. The response is constrained to the `PageVisuals` schema (`visuals[{type, title, summary, data[{label, value}]}]`).
5. Calls run in a thread pool of `VISION_CONCURRENCY` (3); the rate limiter still paces them. A failed page adds a warning and never fails the upload.
6. `serialize` turns each visual into text: `[diagram: Progress toward carbon neutral] … Values: Baseline emissions7 = 445 kg; Emissions reductions = -99 kg; iMac emissions10 = 346 kg`.

Bumping `PROMPT_VERSION` invalidates the cache when the prompt changes.

---

## 10. Ingest stage 6 — Chunking ([chunking.py](backend/app/chunking.py))

### 10.1 Reading-order items — `build_items`

For each non-skipped page, in reading order:

- A line that anchors a table → a **table** item (the `Table` object).
- A `FIGURE`/`CHART`/`MAP` caption → a **figure** item: the caption, its continuation lines, and the small-font lines that follow (panel titles, notes, sources). Near-letterless debris like axis ticks is dropped.
- Headings, box markers, table cells and footnote lines end the current paragraph and are not emitted. Headings live in the section path instead.
- Body lines of the same PDF block and the same section form a **text** item (`join_lines` de-hyphenates).

Then **continuation merge**: a text item that doesn't end in terminal punctuation, followed by one in the same section that starts lowercase, is the same paragraph interrupted by a column or page break, and the two are joined. Finally, text items with fewer than 3 letters are dropped. The threshold is that low on purpose, so short labelled values like `• Production: 50%` survive.

### 10.2 Packing children — `Chunker.pack_run`

Consecutive text items **of one section** form a run, so a chunk never crosses a section boundary. Within a run:

- Items over `CHUNK_MAX` (1400) are split into sentences.
- Pieces are packed into chunks. A new chunk starts when adding the next piece would exceed `CHUNK_MAX`, or the chunk already holds `CHUNK_TARGET` (900) characters and the next piece would push past it.
- **Overlap:** a new chunk starts with the previous chunk's last sentence, if that sentence is at most 250 characters, so no chunk begins mid-thought.
- A chunk spans `page`…`page_end`, and collects the footnote refs of its items.

### 10.3 Parents — `Chunker._parents`

Consecutive children of a run are grouped into **parent windows** of up to `PARENT_MAX` (3500) characters. A parent stores the children's text without the overlap sentences, page range, section path and child ids. Retrieval matches the small children; context assembly can swap in the parent ("small-to-big", §16).

### 10.4 Other chunk kinds

- **table:** split by rows so each markdown piece stays at most 2,500 characters, with the caption and header repeated. `text` = markdown, embedded text = caption + subtitle + row sentences.
- **figure:** caption + description.
- **visual:** each chart transcription, prefixed with "Chart/infographic transcribed from page N", placed in the section of the page it describes.
- **footnote:** every note as its own chunk: `Footnote 18: … (Cited on page 7.)`. "What assumptions are behind the 58% figure?" can then hit the note directly.

### 10.5 Contextual header and footnote linking — `Chunker.add`

Every chunk gets:

```text
header      = "<doc title> › <last 3 path elements, each with aliases> · p.<printed label>[ · table|figure|chart transcription|footnote]"
              e.g. "Global Economic Prospects — June 2026 › Chapter 2. Regional Outlooks (Ch. 2) › South Asia (SAR) › Outlook · p.74"
text        = chunk text + "\nNotes: [18] Energy consumption … 143W Power Adapter …"   (≤4 notes, ≤450 chars each)
search_text = header + "\n" + text        (or table row sentences instead of text)
```

The **header is embedded with the chunk**. This single step is what lets "South Asia" find a chunk whose text only says "SAR", and what tells GEP's seven near-identical "Outlook" subsections apart. `CONTEXT_HEADERS=off` removes it for ablation.

Chunk record fields: `id, doc_id, file_name, kind, page, page_end, page_label, section_id, section_path, parent_id, order, footnotes, header, text, search_text`.

---

## 11. Ingest stage 7 — Embedding and storage ([store.py](backend/app/store.py))

`RAG.index_pdf` embeds every chunk's `search_text` (via the cache) and calls `Store.add_document`.

### 11.1 On disk

```text
data/index/meta.json       version, embedding namespace, document records   ← written last
data/index/chunks.jsonl    one chunk per line
data/index/parents.json    parent windows
data/index/vectors.npy     float32 matrix, row i ↔ chunk i
```

Every file is written to a temp file and swapped in with `os.replace`. `meta.json` goes last and marks the index as complete. On load, a version mismatch, missing file or vector/chunk count mismatch resets to an empty index instead of serving corrupt data. A Phase 1 `index.json` is detected and reported (`/api/stats` warns to re-upload); it is never mixed in.

### 11.2 Safety rules in `add_document`

- **Embedding namespace check.** Vectors from different models are not comparable. If the index was built with another provider or model, adding is refused with "clear the knowledge base before switching".
- **Replacement on re-upload.** Existing chunks with the same `doc_id` (content hash) *or* the same file name are removed first. Duplicates would otherwise crowd the top-k (a Phase 1 lesson).

### 11.3 BM25 — keyword search

The tokenizer lowercases and keeps numbers whole (`6.3`, `802,834`, `45%`). It drops stopwords (including "according", "report", "document", "say"). Light stemming applies to alphabetic tokens over 4 characters: `-ies` → `-y`, trailing `-s` dropped (not `-ss`). `BM25` keeps postings per term and scores with k1 = 1.5, b = 0.75. It is rebuilt in memory on load, which takes milliseconds.

### 11.4 Search primitives

- `mask(doc_ids, chunk_filter)`: a boolean row mask for scoping.
- `dense(qvec, mask, k)`: `matrix @ qvec` (vectors are normalised, so this is cosine), masked rows set to −∞, top k.
- `sparse(tokens, mask, k)`: BM25 scores, masked, top k with score > 0.

A re-entrant lock serialises every mutation and query, so a background ingest can't corrupt a concurrent search.

---

## 12. Answering stage 1 — Routing ([rag.py](backend/app/rag.py))

`RAG.ask(question, doc_ids, debug)` saves the user turn, runs `_route`, saves the assistant turn. `_route` is an ordered decision tree:

| # | Condition | Response |
|---|---|---|
| 1 | "what's my name / who am I" | from memory, no model call |
| 2 | "my name is X / call me X / I'm X" (X not in a list of non-names like "looking", "trying") | store the name, acknowledge |
| 3 | greeting or small talk ("hi", "thanks", …, whole message) | conversational prompt, no retrieval |
| 4 | index empty | conversational prompt ("no documents uploaded") |
| 5–7 | retrieve (§13–16); refused or empty | fixed "couldn't find it" message |
| 8 | the question names an identifier (`ID 006`, `record 42`) that no retrieved chunk contains | refuse rather than answer from a neighbouring record |
| 9 | otherwise | grounded answer + sources + grounding report |

Order matters: memory and small talk never touch retrieval, so "hi" can't be answered from a document.

---

## 13. Answering stage 2 — Query analysis ([retrieval.py](backend/app/retrieval.py))

`analyse(store, question, doc_ids)` builds a `QueryPlan` with no model call.

### 13.1 Intent

- **summary**: "what does … say", "summar*", "overview", "main risks/findings", "describe", "explain", …
- **numeric**: "how much/many", "what percentage/share", "total", "rate", "forecast", "projected", …

Intent controls how many sources are kept and whether parent windows are used.

### 13.2 Document scope — `_route_document`

The UI selection wins. Otherwise, with 2+ documents, each question token votes for a document if it is **nearly exclusive** to it: it appears in at least 2 chunks of the top document and in at most `max(1, top/20)` chunks of all others. "iMac", "GPO", "FDsys", "asia" vote; "growth" appears everywhere and doesn't. One document with votes, or one with at least twice the votes of the runner-up, becomes the scope. The reason is recorded for debugging (`terms only in this document: asia, south`).

### 13.3 Hard filters: explicit structure

| Pattern | Filter |
|---|---|
| `chapter 3` / `ch. 3` | chunks whose path contains a section titled "Chapter 3…" |
| `box 1.1` | chunks under "Box 1.1 …" |
| `executive summary`, `statistical annex`, `definitions`, `endnotes` | chunks under that section |
| `table 2.10` / `figure 1.3` | table, figure or visual chunks whose first 200 characters contain the label |
| `footnote 18` | that footnote chunk |

A filter is applied only if it matches something, so a reference to a non-existent chapter can't empty the results.

### 13.4 Soft boosts: named sections

A section title of 2+ words, at most 70 characters and not generic ("Outlook", "Recent developments", …) that appears in the question as a phrase is a soft section: "South Asia", "East Asia and Pacific". An uppercase alias in the question ("SAR") also selects its section. Soft sections add extra ranked lists (§14); they don't filter.

### 13.5 Multi-part questions — `split_parts`

"What new digital system replaced FDsys, **and when** was it fully deployed?" is split at `, and` followed by a question word (`when|how|what|which|where|why|who`) into sub-questions. Each part is searched on its own (dense + BM25 lists at full weight in the fusion, §14). Later parts carry the first part as context, so "when was *it* deployed" still knows what "it" is. The parts are also listed in the prompt as a checklist the model must answer in order. Without this, the model tended to answer only the first half, even when the passage for the second half was retrieved.

### 13.6 Acronym expansion

Using the scoped documents' glossaries: if the question contains an expansion ("South Asia"), the acronym's tokens (`sar`) are added to the **BM25** query; if it contains the acronym, the expansion's tokens are added. The dense query stays natural language, since embeddings handle paraphrase themselves.

---

## 14. Answering stage 3 — Hybrid search and fusion

`search(store, llm, question, plan)`:

1. **Mask** = document scope ∩ hard filter (if it matches anything).
2. **Ranked lists**, each of `CANDIDATES` (40):
   - `dense`: cosine against the question embedding;
   - `bm25`: keyword scores over `tokenize(question) + bm25_extra` (if `HYBRID`);
   - `dense@section` and `bm25@section`: the same, restricted to soft sections, 20 each, **weight 1.5**.
3. **Reciprocal Rank Fusion:** `score(chunk) = Σ weight / (60 + rank)`. RRF needs no score calibration, which matters because cosine and BM25 scores live on different scales. A chunk ranked well by several lists rises.
4. **Diversity**, in fused order:
   - drop a chunk whose token set overlaps an already kept chunk with Jaccard ≥ 0.8 (GPO states "166 depository libraries" on p5 *and* p21);
   - keep at most 3 chunks per page;
   - stop at `max(RERANK_TOP, TOP_K)` candidates.

Why hybrid: embeddings blur exact tokens (`FDsys`, `ENERGY STAR`, `802,834`, `Table 2.10`), and BM25 misses paraphrase. Each catches what the other misses.

---

## 15. Answering stage 4 — Reranking and the relevance gate

`rerank` sends the top `RERANK_TOP` (20) candidates to Gemini in one JSON call. Each passage is shown as `[n] header` plus its **best window**: the ~900-character span of the chunk with the most question-token overlap (`best_window`). A plain prefix would hide answers that sit late in a chunk, as with the runbook's rollback command at character ~750.

The model scores every passage 0–3: 3 directly answers, 2 partially answers or gives needed context, 1 same topic but unhelpful, 0 irrelevant. A passage about a different region, chapter, year or configuration than the one asked about can score at most 1.

**The gate:**

- No passage scored at least 2 → refuse ("couldn't find it"). This is how out-of-scope questions are refused: similarity search always returns *something*, but a reranker can say "none of this answers it".
- Otherwise keep passages with score ≥ 2, sorted by score (ties keep fused order). Keep up to `TOP_K` (8) for summary or numeric questions, and up to 6 for others.
- If the rerank call fails, or `RERANK=off`, fall back to fused order with the cosine gate: the best dense score must reach `RELEVANCE_THRESHOLD`.

---

## 16. Answering stage 5 — Context assembly

`assemble(store, selected, plan)`:

1. **Small-to-big.** For a summary question, or when 2+ selected children share a parent, the parent window replaces those children (once). A summary needs the whole passage, not one fragment.
2. **Budget.** Blocks are added in rank order up to `CONTEXT_CHAR_BUDGET` (12,000 characters). The first block is always kept.
3. **Reading order.** The blocks are sorted by document, then by `order`, so the model reads a coherent narrative rather than a score-ordered jumble.

Each block is labelled for the model:

```text
[SOURCE 2] GEP-Jun-2026.pdf · Chapter 2. Regional Outlooks › South Asia › Outlook · p.74 (PDF p.96) · text
…text…
```

---

## 17. Answering stage 6 — Generation and the grounding check

### 17.1 The grounded prompt (`GROUNDED_PROMPT`)

The rules tell the model to:

- use only the sources;
- give exact figures with units and the period, region or configuration they apply to;
- use only the region, chapter, year or configuration that was asked about, and name it;
- report a disagreement between sources (e.g. a table and a footnote) rather than pick one;
- say when a figure came from a chart transcription;
- say plainly when the sources don't answer;
- cite `[SOURCE n]`.

A remembered user name may be used for tone but is explicitly "NOT document evidence". Generation runs at temperature 0.2 (rerank and vision at 0).

### 17.2 Numeric grounding — `grounding.unsupported_numbers`

Wrong numbers are the most damaging hallucination here, and the cheapest to detect. Every number in the answer (after removing `[SOURCE n]` markers and list numbering) must be supported by a number in the context. A number counts as supported when:

- it exactly equals a context number;
- it is the second half of a year range (`2027–28`);
- it matches within rounding of its shown decimals, at any scale of 1, 10³, 10⁶ or 10⁹ in either direction. So `874.5 million` ≈ `$874,466,000`, and `$874,466,000` ≈ `874,466` in a table "in thousands".

If any number is unsupported, the answer is regenerated **once**, with a note listing the offending figures. The retry is kept if it has no more unsupported figures than the original. Any that remain are returned in `grounding.unsupported`, and the UI shows a warning chip.

---

## 18. Memory

`Memory` ([rag.py](backend/app/rag.py)) is a JSON file with `facts` (currently the user's name) and `history` (the last 12 turns; the conversational prompt uses the last 10). It is deliberately separate from the document index:

- clearing documents keeps memory, and vice versa;
- memory never enters the grounded prompt as evidence;
- the conversational prompt uses it for tone and continuity only.

---

## 19. Background ingestion jobs ([jobs.py](backend/app/jobs.py))

Indexing GEP takes minutes, longer than a browser should hold an upload request open. `JobManager` keeps:

- a **queue** and a single daemon worker thread. Uploads are indexed one at a time, in order, and the ingest lock in `RAG` serialises them anyway.
- a **registry** of job dicts: `id, file_name, status (queued|running|done|error), stage, stage_label, done, total, chunks, error, warnings, created, finished`. It is bounded to the newest 50 jobs.

`RAG.index_pdf` reports progress through a callback at each stage: parsing, layout, vision (n/N), chunking, embedding (n/N), saving. The worker copies this into the job. `wait(job_id)` lets `?wait=true` uploads and scripts block until completion.

---

## 20. HTTP API ([main.py](backend/app/main.py))

The `RAG` engine is created lazily on first use, so `/api/health` can report key status without touching Gemini. Without a key, every engine route returns **503** with instructions.

| Route | Behaviour |
|---|---|
| `GET /` | serves `frontend/index.html` with no-cache headers |
| `GET /api/health` | status, key present, chat model, embedding provider/model, vision mode, rerank mode |
| `POST /api/upload` | validates every file first: has a name, `.pdf`, ≤ `MAX_FILE_SIZE_MB`, non-empty, starts with `%PDF` (any failure → 400, nothing queued). Then queues a job per file → `{jobs}`. `?wait=true` → `{indexed, stats}` or 400 with the indexing error. |
| `GET /api/jobs/{id}` | job status (404 if unknown) |
| `POST /api/ask` | `{question (1–2000 chars), doc_ids?, debug?}` → answer. 422 on bad input, 500 on model failure. |
| `GET /api/stats` | counts by kind, file names, legacy-index warning |
| `GET /api/documents` | document records (glossary and aliases omitted) + chunk counts by kind |
| `DELETE /api/documents/{doc_id}` | remove one document (404 if unknown) |
| `DELETE /api/knowledge-base` | clear all documents, keep memory |
| `DELETE /api/memory` | clear memory, keep documents |

---

## 21. Frontend ([frontend/index.html](frontend/index.html))

A single self-contained file: no build step, no framework, no CDN.

- **Upload:** drag-and-drop or file picker → `POST /api/upload`. Each returned job is polled every 0.9 s (`trackJob`), and its toast shows the live stage ("Embedding 640/830"). Done → success toast plus any warnings. Error → error toast.
- **Documents:** `refreshDocuments` lists each document with its profile badge, page and chunk counts, and counts of tables, charts and footnotes. A remove button sits on each.
- **Scope selector** (shown with 2+ documents): sends `doc_ids` with each question.
- **Answers:** an escape-first markdown renderer (code, bold, bullets, paragraphs). The sources panel (`sourceRow`) shows the file, printed and PDF page, a kind badge (text/table/chart/figure/footnote), the section path, an excerpt, and the rerank relevance (n/3). If grounding flagged figures, a red "Check: … not found in the sources" chip appears.
- Dark and light themes, a responsive drawer below 860px, a compact header below 480px, keyboard shortcuts (`Enter`, `Shift+Enter`, `/`), and reduced-motion support.

---

## 22. Testing and evaluation

### 22.1 Offline tests (no API key)

- [tests/test_pipeline.py](backend/tests/test_pipeline.py) (50 checks). The full pipeline over the Phase 1 sample PDFs, with a `FakeLLM`: bag-of-words embeddings, a word-overlap "reranker", canned answers. It covers indexing, restart persistence, re-upload replacement, bad-PDF rejection, every routing branch, the ID guardrail, rerank refusal, the cosine fallback gate, the grounding retry, and memory separation.
- [tests/test_documents.py](backend/tests/test_documents.py) (67 checks). Each document finding is pinned on the real Phase 2 PDFs: ligature repair, footer removal, drop cap, page labels (`p.74`, `xvii`, `c1`), contents and abbreviations skipping, glossary, South Asia snapped to p96 under Chapter 2 with alias SAR, Box 1.1 extent, Table 2.10 (Maldives 2026f = 0.7), GPO balance sheet columns, iMac power table (sleep at 115V = 0.57W), 19 endnotes with note 18 linked to p7, key/value joins, grounding rules, and query analysis (routing, filters, boosts, acronym expansion).

### 22.2 Live evaluation ([backend/eval/](backend/eval/))

- [questions.json](backend/eval/questions.json): 34 questions, namely the 9 sample questions, 20 held-out questions (numeric, table, chart, footnote, summary, multi-part), 2 traps and 3 refusals. Each has regexes that must or must not match, expected pages and optionally an expected section. They were written before the pipeline work and never edited to make a run pass.
- [run_eval.py](backend/eval/run_eval.py) indexes into its own work directory (`eval/.work`), asks every question with `debug=true`, and grades it:
  - **pass:** all `must_contain` match, no `must_not_contain` matches, and it isn't a refusal. For negatives, a refusal *is* the pass.
  - **strict pass:** pass *and* an expected page is in the context. This catches right-answer-from-wrong-section.
  - **hit@k, MRR, section precision, unsupported numbers, latency.**
  - Reports are written as `eval/results/<time>-<label>.md/.json`. `--set KEY=VALUE` overrides config (ablation); `--pipeline phase1` runs the frozen Phase 1 engine ([phase1_rag.py](backend/eval/phase1_rag.py)) with the same embedder and model, so baseline and Phase 2 differ only in pipeline.
- [run_ablations.sh](backend/eval/run_ablations.sh) switches one stage off at a time.

Results and analysis: [plans/phase 2/EVAL-RESULTS.md](plans/phase%202/EVAL-RESULTS.md).

---

## 23. Walkthroughs: three questions end to end

### "What is the growth outlook for South Asia in this report?"

1. Routing: not memory or small talk → retrieve.
2. Query analysis: "asia"/"south" occur only in GEP → scope = GEP. "South Asia" is a section title → soft section. The glossary has `SAR → South Asia` → BM25 adds `sar`. No intent flags.
3. Search: dense and BM25 over GEP, plus both inside the South Asia section at weight 1.5. The p96 chunk ranks high in all four lists, because its embedded header reads "… › South Asia (SAR) › Outlook · p.74" and its text contains "SAR" and "6.3 percent".
4. Rerank: the p96 outlook chunk and Tables 2.9/2.10 score 3. Other regions' outlooks score at most 1 under the "different region" rule.
5. Context in reading order: p96 text, then the p98 tables.
6. Answer: "soften to 6.3 percent in 2026 … pick up to an average of 7 percent per year in 2027–28 [SOURCE 1]". Every figure appears in the sources, so no retry.

### "What were the iMac's baseline emissions before reductions?"

1. Scope: "imac" is exclusive → iMac report.
2. Search: the **visual** chunk from p2 ("Baseline emissions7 = 445 kg; Emissions reductions = -99 kg; iMac emissions10 = 346 kg") matches strongly. It only exists because vision transcribed the waterfall chart at ingest.
3. Rerank 3 → the answer reports 445 kg and −99 kg, marked as read from the chart.

### "What does the report say about the FIFA World Cup?"

1. No exclusive terms → all documents searched.
2. Dense and BM25 return the "closest" chunks, as they always do.
3. Rerank: everything scores 0–1 → the gate refuses → "I couldn't find anything in the uploaded documents…". No model answer is generated at all.

---

## 24. Failure modes and how each is handled

| Failure | Handling |
|---|---|
| Free-tier 429 per minute | client-side pacing; retry with the server's suggested delay |
| Daily quota exhausted | `QuotaExhausted` with advice (`EMBEDDING_PROVIDER=local`); the job fails with that message |
| 5xx / deadline from Gemini | up to 6 retries, exponential backoff with jitter |
| Vision call fails on one page | warning on the job; ingest continues |
| Reranker call fails | falls back to fused order + cosine gate |
| Non-PDF / empty / oversized / not `%PDF` | 400 before anything is queued |
| Scanned PDF (no text) | "may need OCR" error |
| Password-protected PDF | "password-protected" error |
| Crash mid-save | atomic writes; `meta.json` last; an incomplete index loads as empty instead of corrupt |
| Index from another embedding model | add refused until the knowledge base is cleared |
| Phase 1 index present | ignored and reported; re-upload needed |
| Same file uploaded twice | old chunks replaced, not duplicated |
| Model invents a figure | numeric check → one regeneration → remaining figures flagged in the UI |
| Question about a missing record ID | exact-ID guardrail refuses |

---

## 25. Extending the system

- **A new document type** (e.g. DOCX): produce the same `ParsedDoc` (pages of styled `Line`s) and every later stage works unchanged.
- **Better layout analysis:** replace heuristics in `tables.py`/`structure.py` with a layout model (Docling, DocLayout-YOLO) behind the same `Table`/role interfaces. `tests/test_documents.py` tells you immediately whether the findings still hold.
- **A cross-encoder reranker:** implement `rerank()` with a local model returning 0–3-like scores; the gate logic stays.
- **A vector database:** `Store.dense/sparse/mask` are the only search primitives the retriever uses. Back them with pgvector, Qdrant or OpenSearch.
- **New evaluation questions:** add to `eval/questions.json` with `must_contain` regexes and `expected_pages`, then run `run_eval.py`. Write them *before* changing the pipeline.
