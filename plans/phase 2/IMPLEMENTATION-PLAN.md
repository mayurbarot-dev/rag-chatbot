# Phase 2 Implementation Plan — Advanced RAG over Three Document Shapes

**Brief:** [RAG-Hands-On-Use-Cases.docx](RAG-Hands-On-Use-Cases.docx) and [row-detail.md.txt](row-detail.md.txt).
**Starting point:** the Phase 1 chatbot ([plans/phase 1/](../phase%201/)). It uses FastAPI, pypdf, fixed 1000-char chunks, Gemini embeddings, a JSON vector store, cosine top-5 retrieval and a grounded prompt.

**Goal:** one pipeline that correctly answers the nine sample questions plus a harder set of held-out questions across all three documents. Every change has to be justified by a measured improvement, not by intuition.

> **Status (2026-10-01): implemented, milestones M0–M8.** Results: [EVAL-RESULTS.md](EVAL-RESULTS.md). The Phase 1 baseline scored 23/28; Phase 2 scores 28/28, with ablations for each stage. How to test: [HOW-TO-TEST.md](HOW-TO-TEST.md). How it works: [TECHNICAL-GUIDE.md](../../TECHNICAL-GUIDE.md).
>
> **Where the build deviates from this plan, and why:**
> - **Embeddings default to a local model** (`BAAI/bge-base-en-v1.5` via fastembed), not Gemini at 768 dimensions. The free Gemini tier capped embeddings at 100 texts/minute plus a daily limit, and that stopped the very first baseline run. Gemini embeddings remain one setting away (`EMBEDDING_PROVIDER=gemini`), behind a persistent cache and pacing.
> - **Document routing uses terms exclusive to one document**, not LLM-written doc cards. It is deterministic and costs no model call; it routed every eval question correctly.
> - **Near-duplicate suppression happens at query time** (Jaccard ≥ 0.8 among candidates), not at index time. The effect is the same, without an O(n²) pass over the index.
> - **The `PARSER`/`CHUNKING` ablation switches were replaced** by running the frozen Phase 1 engine as the baseline (`--pipeline phase1`). That baseline is pypdf + fixed chunking.
> - **Table rebuilding needed more cases than planned:** annotation rows, header fragments on the caption's baseline, centred titles, and continuation tables. Heading detection uses **per-page** body style, because GPO mixes two body fonts.

```text
Document → Parsing → Structure → Chunking → Embedding → Retrieval → Reranking → Context → LLM → Evaluation
```

---

## 0. What the documents actually look like (measured, not assumed)

Before planning, each PDF was run through the current Phase 1 parser (pypdf) and through PyMuPDF. These findings drive every decision below.

### GPO 2018 Annual Report — 82 pages, narrative (Use Case 1)

| Finding | Evidence | Consequence |
|---|---|---|
| All three answers exist as plain sentences | p5: *"GPO staff visited 166 depository libraries in 25 states"*; p4: *"full-scale deployment of govinfo … successor to our FDsys website … retiring FDsys in December 2018"*; p5: *"GPO procured more than $375 million worth of printing…"* | Baseline case. Fixed chunking probably already passes, and M0 will confirm it |
| The same facts repeat in several places | 166 libraries on p5 **and** p21; $375.7M in a stats table on p14 | Near-duplicate chunks will compete for top-k, so dedupe/diversify is needed |
| Pages 35–82 are audited financial statements with tables | The PDF outline has 57 entries, all inside the financial section | Narrative and tabular content are mixed in one file |
| Running header is letter-spaced | `R E P O R T \| U . S . G O V E R N M E N T …` | Pollutes chunks, so it must be stripped |
| pypdf garbles apostrophes | `GPO�s` (pypdf warns that `fontTools` is missing). PyMuPDF yields `GPO’s` correctly | Parser choice matters |
| PDF page ≠ printed page | Page labels `c1, c2, 1, 2, …` | Citations should show the printed page |

### World Bank Global Economic Prospects, June 2026 — 200 pages, structured (Use Case 2)

| Finding | Evidence | Consequence |
|---|---|---|
| A real PDF outline exists (18 entries) | Exec Summary p19, Ch.1 p23, Ch.2 p69, East Asia & Pacific p81, Europe & Central Asia p84, LAC p88, MNA p92, South Asia p95, SSA p99, Ch.3 p107, Ch.4 p143, Statistical Annex p185, plus Boxes 1.1/1.2/2.1/3.1 | A free section tree, though it is not fully trustworthy (next row) |
| **Outline destinations are off by one** | Outline says South Asia → p95, but p95 is the MNA forecast table. The `South Asia` heading is at the top of p96 | Use the outline as a hint, then confirm against detected headings |
| **Region text uses acronyms, not names** | *"Growth in **SAR** is expected to soften to 6.3 percent in 2026"* (p96). The question asks about "South Asia" | Pure embedding similarity is weak here. Section labels and the acronym glossary are needed |
| An abbreviations page exists | p21: `SAR = South Asia`, `SSA = Sub-Saharan Africa`, `EAP = …` | Parse it into a glossary for query expansion |
| **"Th" ligature is broken in both parsers** | pypdf: `7T_he global economy`. PyMuPDF: `Te global economy` (186 occurrences) | A text-repair step is needed. Also `ﬁ`/`ﬂ` ligatures need NFKC |
| Front matter pollutes retrieval | Contents / list of figures / list of tables (p7–p13) with dot leaders: `2.9 South Asia forecast summary ..... 76` | These would out-score real content for "South Asia forecast", so exclude them |
| Two-column body plus running header | `CHAPTER 2 GLOBAL ECONOMIC PROSPECTS \| JUNE 2026 74` | Strip it, but harvest the chapter and the printed page (`74`) from it first |
| Dense numeric tables come out one span per cell | Table 2.8 / 2.10 cells extracted as separate spans aligned on y. PyMuPDF `find_tables()` **mangles** them (wrong headers) | Rebuild rows from coordinates ourselves. Don't trust `find_tables()` |
| Hyphenated line breaks | `confi-\ndence` | De-hyphenate during repair |
| 23 near-empty pages (covers, dividers, figure-only pages) | p1–5, 8, 14, 16, 22–24, 69–70, 106–108, … | Figure-only pages are candidates for vision |

### Apple iMac Product Environmental Report — 12 pages (not 11), visual (Use Case 3)

| Finding | Evidence | Consequence |
|---|---|---|
| ENERGY STAR answer is in text | p7: *"iMac uses 58 percent less energy than the requirement for ENERGY STAR.18"* | Easy to retrieve, **but** the marker `18` is fused onto the sentence and footnote 18 (on p12) defines the test conditions |
| Carbon footprint is a flattened table | p10: `Total product footprint 346 kg CO2e … Production 50% · Transportation 5% · Product use 45% · End-of-life <1%`, plus a second configuration (four ports, 512GB) at **391 kg** | Answers must name the configuration. Two numbers must not be blended |
| **The source contradicts itself** | The p10 table says 346 kg is for *two ports 256GB*. Footnote 10 says the calculation is *"based on iMac (two ports) with 512GB"* | A grounded answer should report the table and flag the footnote. This makes a good hallucination test |
| The p2 waterfall chart is unreadable as text | The labels come out as a jumble: `–99 kg`, `346 kg`, `445 kg Baseline emissions7`, repeated `Electricity for manufacturing` | Only a vision pass recovers "baseline 445 kg − 99 kg reductions = 346 kg" |
| Stand answer appears 3× | p1 icon call-out, p2, p4: *"100 percent recycled aluminum in the stand"* | Easy, so this is a regression guard |
| Power table is glued onto the endnotes | p12 rows `Sleep 0.56W 0.57W 0.59W` come after footnote 19, and the header `100V 115V 230V` is on its own line | Rebuild it as a table and keep it separate from the footnote text |
| Footnote markers are machine-detectable | PyMuPDF span flag bit 0 (superscript): 36 markers found, e.g. p7 `18`, p10 `7` | Footnotes can be linked to the claims they qualify, with no guessing |

### Expected answers (the pass/fail key)

| # | Doc | Question | Expected answer | Where |
|---|---|---|---|---|
| G1 | GPO | Depository libraries visited in FY 2018? | **166** libraries in 25 states (3-year total > 610) | p5, p21 |
| G2 | GPO | What replaced FDsys, and when was it fully deployed? | **govinfo**, full-scale deployment during **FY 2018**; FDsys retirement goal December 2018 | p4, p8 |
| G3 | GPO | Procurement for federal agencies in FY 2018? | **More than $375 million** ($375.7M; > 84,000 orders) | p5, p14 |
| W1 | GEP | Growth outlook for South Asia? | SAR growth to soften to **6.3% in 2026**, picking up to **~7% a year in 2027–28**. Drivers: Middle East conflict, energy prices, remittances, tourism | p96–98 |
| W2 | GEP | What does Chapter 3 say about sovereign debt in developing economies? | Synthesis of Ch.3 *"A Rising Challenge: Sovereign Debt Levels and Interest Rates in EMDEs"*: rising government debt and interest rates | p107–142 (summary p19) |
| W3 | GEP | Exec summary: main risks to the global outlook? | Downside risks: further energy-price surge, sharper inflation, financial stress, conflict escalation, trade disruption, etc. Must come **from the Exec Summary**, p19–20 | p19–20 |
| A1 | iMac | Energy vs ENERGY STAR requirement? | **58% less** (footnote 18: based on ENERGY STAR max allowance, 143W adapter) | p7, p12 |
| A2 | iMac | Total footprint, use vs manufacturing? | **346 kg CO2e** (two ports, 256GB): **production 50%**, **product use 45%**, transport 5%, EoL <1%. Four-port 512GB: 391 kg | p10 (p2 chart) |
| A3 | iMac | Stand material and % recycled? | **Aluminum, 100% recycled** | p1, p2, p4 |

---

## 1. Design principles

1. **Measure first.** Build the evaluation harness before touching the pipeline, record the Phase 1 baseline, and require every milestone to move a number.
2. **One pipeline, document-aware behaviour.** No `if file == "GEP"` branches. Each document gets a **profile** computed at ingest: page count, outline present, visual density, table density. The profile turns on structural chunking, table rebuilding, footnote linking and vision as needed. This is the point of the exercise: the same stages, tuned by the document's shape.
3. **Structure is metadata, not just text.** Every chunk knows its document, section path, page, printed page label and content kind (`text | table | figure | footnote | visual`). Retrieval, filtering, context assembly and citations all use it.
4. **The generator is a dumb reader of trustworthy context.** Most "hallucinations" here come from the wrong chunk (wrong region, wrong config, TOC page), not from the model. Fix retrieval before prompting.
5. **Keep Phase 1 guarantees.** The memory/evidence separation, casual-message routing, the exact-ID guardrail, re-upload replacement, the stubbed offline test suite and the single-file frontend stay.

---

## 2. Target architecture

### 2.1 Module layout

`rag.py` (557 lines) is split along pipeline stages. Each module is small and unit-testable without an API key.

```text
backend/app/
├── config.py        + new settings (§9)
├── llm.py           Gemini wrapper: embed (batched, output_dimensionality), generate,
│                    generate_json (response_schema), generate_with_image; retry + backoff
├── parsing.py       PyMuPDF layout extraction → Page[Block]; text repair; header/footer
│                    removal; page labels; front-matter detection
├── structure.py     Section tree from outline + detected headings; glossary extraction;
│                    document profile
├── tables.py        Coordinate-based table reconstruction + serialization
├── footnotes.py     Superscript marker detection, footnote parsing, linking
├── vision.py        Visual-page selection, page rendering, Gemini transcription, cache
├── chunking.py      Structure-aware chunking, parent/child, contextual headers
├── store.py         Vector store (npy + jsonl) + BM25 index, atomic writes, versioning
├── retrieval.py     Query analysis, scoping, hybrid search, RRF, rerank, context assembly
├── grounding.py     Numeric faithfulness check, citation handling
├── jobs.py          Background ingestion jobs with progress
├── rag.py           Orchestrator: routing tree, memory, answer (slimmed down)
└── main.py          + job endpoints, document list, scoped ask
backend/eval/
├── questions.json   Sample + held-out + adversarial questions with expected facts/pages
├── run_eval.py      Runs the set against the live pipeline, with per-stage toggles
└── results/         Timestamped reports (gitignored except the final one)
backend/tests/
├── test_pipeline.py (updated) + test_parsing.py, test_structure.py, test_tables.py,
    test_footnotes.py, test_chunking.py, test_retrieval.py, test_grounding.py
```

### 2.2 Chunk record

```json
{
  "id": "gep-3f9a1c:c0412",
  "doc_id": "gep-3f9a1c",                // short sha256 of file bytes
  "file_name": "GEP-Jun-2026.pdf",
  "kind": "text",                        // text | table | figure | footnote | visual
  "page": 96, "page_end": 96,            // PDF page indices (1-based)
  "page_label": "74",                    // printed page number
  "section_path": ["Chapter 2. Regional Outlooks", "South Asia", "Outlook"],
  "section_id": "s-2.6.2",
  "parent_id": "gep-3f9a1c:p0098",       // larger context window this child belongs to
  "order": 412,                          // reading order within the document
  "footnotes": [],                       // linked footnote numbers
  "header": "GEP June 2026 › Ch 2 Regional Outlooks › South Asia (SAR) › Outlook · p.74",
  "text": "Growth in SAR is expected to soften to 6.3 percent in 2026 …"
}
```

The **embedded and BM25-indexed string** is `header + "\n" + text` (contextual chunk header). The `text` alone is what the model reads, labelled with the header in the SOURCE block.

---

## 3. Stage-by-stage design

### 3.1 Parsing (`parsing.py`): PyMuPDF replaces pypdf

**Why:** pypdf gives a flat string. We need span-level font size, bold and superscript flags, bounding boxes, drawings/images per page, and page rendering for vision. PyMuPDF gives all of these and handled GPO's apostrophes correctly where pypdf did not.

Steps per page:

1. `page.get_text("dict")` → blocks → lines → spans (text, size, flags, font, bbox).
2. **Text repair** (pure functions, unit-tested):
   - `unicodedata.normalize("NFKC")` fixes `ﬁ ﬂ ﬀ`.
   - **Th-ligature repair:** a closed list of `T`+rest → `Th`+rest words (`Te→The`, `Tis→This`, `Tat→That`, `Tese→These`, `Tey→They`, `Teir→Their`, `Tere→There`, `Ten→Then` *only when* the doc shows the ligature fault, `Tus→Thus`, `Trough→Through`, `Tree→Three`, `Tird→Third`, …). Apply only when the document's `\bTe\b` count is above a threshold, so normal documents are untouched. Also handle the pypdf form `7T_h`.
   - De-hyphenation: `confi-\ndence → confidence` when the joined word is lowercase-continuous. Keep real hyphens (`2027–28`, `low-carbon`).
   - Collapse letter-spaced runs (`R E P O R T` → `REPORT`) before header matching.
3. **Running header/footer removal:** lines in the top/bottom 8% of the page whose normalized form (spaces and digits removed) repeats on ≥ 30% of pages. **Harvest before deleting:** the printed page number (`74`, `xvii`) → `page_label`, and the chapter token (`CHAPTER 2`) → a structure hint.
4. **Front-matter detection:** flag a page `skip` when ≥ 25% of its lines end in dot leaders plus a number (TOC, list of figures, list of tables), or it is a cover/copyright page with under 100 chars. Skipped pages aren't chunked. The TOC is still used as a structure hint.
5. **Abbreviations page:** detect it (a heading "Abbreviations" plus many short uppercase tokens) and parse it into `glossary = {"SAR": "South Asia", …}`. Store it on the document and don't chunk it.
6. **Block classification:** `heading` (size ≥ body + 1.5pt, or bold and short and standalone), `paragraph`, `caption` (`^(FIGURE|TABLE|BOX) \d`), `table_region` (§3.3), `footnote` (§3.4), `figure_only` (images/drawings with little text).
7. **Reading order:** sort blocks by column (x-midpoint clustering into 1–2 columns), then by y. Verify on GEP p96 (two columns) and p19 in tests.

Body font size = the mode of span sizes weighted by characters. Measured: GPO 10/12pt, GEP 11pt, iMac 8pt.

> **Licence note:** PyMuPDF is AGPL-3.0. That is fine for this training project. A closed-source product would need a commercial licence or `pdfplumber` + `pypdfium2` (MIT/BSD), at the cost of speed and some convenience. `pypdf` stays in requirements as a fallback for PDFs PyMuPDF can't open.

### 3.2 Structure (`structure.py`)

**Section tree** built in two passes:

1. **Outline seed:** `doc.get_toc()` → `(level, title, page)`. For GEP this gives chapters, regions and boxes. Titles are cleaned (`" -- South Asia"`, embedded newlines).
2. **Heading reconciliation:** for each outline entry, search detected headings on `page-1 … page+2` for a fuzzy title match (token ratio ≥ 0.8). Snap the section start to that heading's exact position (page + y). This fixes GEP's off-by-one (South Asia → p96 top) and splits mid-page boundaries correctly.
3. **Sub-headings:** detected headings inside a section become children (`Recent developments`, `Outlook`, `Risks` in each GEP region; `Introduction`, `Main findings` in Ch.3).
4. **No outline** (iMac): use detected headings only. iMac's page titles (`Smarter chemistry`, `Design and Source`, `Make`, `Package and Ship`, `Use`, `Recover`, `Definitions`, `Endnotes`) become sections. GPO's narrative part (no outline before p35) gets headings from font size. GPO's financial part keeps its outline.
5. **Section aliases:** each section gets searchable aliases from the glossary, e.g. `South Asia` ↔ `SAR`, `Chapter 3` ↔ `Ch. 3` ↔ the full title. These go into the chunk header.

**Document profile** (stored, shown in the UI):

| Signal | GPO | GEP | iMac |
|---|---|---|---|
| pages | 82 | 200 | 12 |
| outline entries | 57 (financial only) | 18 | 0 |
| visual pages (§3.5) | low | ~45 heavy-drawing | 5 of 12 |
| → profile | `narrative` | `long-structured` | `visual-dense` |

The profile sets defaults: vision `auto` budget, chunk target size, and whether parent windows are page-based or section-based. Every setting is overridable in `.env`.

### 3.3 Tables (`tables.py`)

PyMuPDF's `find_tables()` was tested on all three files and misread headers or split rows (GEP Table 2.10, iMac p12). So tables are rebuilt from span coordinates:

1. **Detect a table region:** starting at a `TABLE n.n` caption (GEP), or a run of ≥ 3 consecutive lines with ≥ 3 spans each whose x-positions align to a shared column grid (iMac p10/p12, GPO financial statements).
2. **Rows:** cluster spans by baseline y (tolerance about 0.4 × font size).
3. **Columns:** column boundaries come from the header row(s) (`2023 2024 2025e 2026f 2027f 2028f`), or from x-gaps if there is no header. Assign each cell span to the nearest column centre.
4. **Header stack:** support two header rows (`Percentage-point differences from January 2026 projections` over `2026f 2027f`) by concatenating parent and child labels. Support section rows (`Calendar-year basis`, `Fiscal-year basis`) as row-group labels.
5. **Serialize twice:**
   - **Markdown table** with caption and units line (`Real GDP growth at market prices in percent`) for the LLM context.
   - **Row sentences** for retrieval: `Table 2.10 South Asia economy forecasts — Maldives: 2023 4.9, 2024 3.5, 2025e 5.7, 2026f 0.7, 2027f 7.2 (percent real GDP growth).`
6. **Chunking:** one chunk per table if it fits in about 2,500 chars. Otherwise split by row groups with the caption and header repeated in every piece. `kind = "table"`.
7. **Footnote superscripts inside cells** (`Iraq 1`, `Lebanon 2`) are stripped from labels and recorded as linked footnotes.

### 3.4 Footnotes (`footnotes.py`)

1. **Markers:** spans with the superscript flag (bit 0) that are numeric (`1`–`99`) or `*†‡`, attached to the preceding word. Remove them from the text (`STAR.18` → `STAR.`) and record them on the block (`footnotes: [18]`).
2. **Definitions:** (a) an `Endnotes` / `Notes` section (iMac p11–12), where each entry is a line starting with a number at small size, continuing until the next number. (b) Page-bottom notes: small-font blocks under a separator line in the bottom zone. (c) GEP table notes below each table.
3. **Linking:** at chunk build time, append the referenced footnote text to the chunk under a `Notes:` line: `Note 18: Energy consumption values are based on the ENERGY STAR Program Requirements for Computers, including the max energy allowance … tested with the 143W Power Adapter…`
4. **Also index each footnote as its own chunk** (`kind = "footnote"`) with a back-reference to the pages that cite it, so "what assumptions are behind the 58% figure?" hits directly.
5. **Separate glued content:** the iMac p12 power table goes through §3.3 instead of being treated as part of footnote 19.

### 3.5 Visual content (`vision.py`)

Charts carry information that text extraction cannot recover (iMac p2 waterfall, GEP figure panels). Gemini is multimodal, so visual pages are transcribed.

**Page selection (`VISION_MODE=auto`):** score each page and pick the highest:

- drawings count > 50 (vector charts), or image area > 25% of the page, **and**
- text extraction quality is poor: few chars per drawing area, or many short repeated label lines (iMac p2: `Electricity for manufacturing` ×3), **or**
- a `FIGURE`/chart caption exists on the page.

Budget: `VISION_MAX_PAGES` per document (default 30), ranked by score. Estimated: iMac all 5 visual pages, GEP its ~30 highest-scoring figure pages, GPO a handful. `VISION_MODE=off|auto|all`.

**Transcription:**

- Render at 150 dpi PNG (`page.get_pixmap(dpi=150)`), and send it with the page's extracted text as a hint.
- The prompt asks for: a chart title, chart type, axis labels/units, **every data label exactly as printed**, the series and their values, call-out icons, and footnote markers. No interpretation, no values that aren't visible, and `unreadable` where unsure. Output uses a fixed JSON schema (`response_schema`), serialized to text.
- Result → `kind = "visual"` chunk(s) on that page, with the section path inherited.
- **Cache:** `data/vision_cache/{sha256(file)}_{page}_{model}_{PROMPT_VERSION}.json`. Re-uploads and re-indexing cost nothing, and evaluation runs are repeatable.
- **Trust:** visual chunks are marked in the SOURCE header (`[transcribed from chart]`). The grounding check (§3.9) treats their numbers as supported only when they also match the page's text-layer labels. Otherwise the answer qualifies them (*"read from the chart on p.2"*).

### 3.6 Chunking (`chunking.py`)

Replaces fixed 1000-char windows.

1. **Never cross a section boundary.** Chunks are built inside a leaf section's blocks, in reading order.
2. **Child chunks (retrieval unit):** pack whole paragraphs to a target of about 900 chars (`CHUNK_TARGET`), max about 1,400. Split oversized paragraphs at sentence boundaries. Overlap is **one sentence**, not 150 chars, so no chunk starts mid-word. Captions attach to the chunk that follows them. Tables, visuals and footnotes are their own chunks (§3.3–3.5).
3. **Parent windows (context unit):** a leaf section, or a contiguous ~3,500-char slice of it for long sections. Retrieval matches children. Context assembly can expand a hit to its parent ("small-to-big"). That matters for W2/W3, which need a whole summary, not one fragment.
4. **Contextual header** (§2.2) is prepended for embedding and BM25. This step is what lets "South Asia" find text that only says "SAR", and what tells apart GEP's seven near-identical "Outlook" subsections.
5. **Page attribution:** chunks spanning a page break carry `page` / `page_end`.
6. **Near-duplicate suppression at index time:** MinHash or normalized-text hash per document. Keep duplicates (GPO p5 vs p21 say the same thing) but mark them `dup_of`, so retrieval diversity (§3.7) can drop one.

Estimated sizes: GPO ~350 chunks, GEP ~1,600, iMac ~60.

### 3.7 Retrieval (`retrieval.py`)

```text
question
  → query analysis (scope, acronyms, intent)
  → candidates: dense top-40  ∪  BM25 top-40   (within scope)
  → Reciprocal Rank Fusion (k=60) → top-20, diversity filter
  → LLM rerank → top-N (adaptive 4–8)
  → context assembly (parent expansion, ordering, budget)
```

**a. Query analysis** (rule-based, no LLM call):

- **Document scope:** an explicit UI selector (new), or a name in the question (`iMac`, `GPO`, `World Bank`, `GEP`, via per-document alias lists derived from title and profile). Otherwise **document routing**: at ingest, each document gets a *doc card* (title + section titles + LLM 3-sentence summary), which is embedded. If the top doc card beats the second by a margin, scope to it. Otherwise search all documents.
- **Section scope:** regexes for `chapter \d`, `box \d\.\d`, `executive summary`, `statistical annex`, `table \d\.\d`, `figure \d\.\d`, plus region names or acronyms matched against section aliases. A match becomes a **hard filter** when the section exists (W2 → Chapter 3, W3 → Executive Summary) and a **boost** otherwise.
- **Acronym expansion:** append glossary expansions both ways (`South Asia` → `+ SAR`) to the BM25 query only. The dense query stays natural.
- **Intent hints:** a number-seeking question (`how much`, `how many`, `what percentage`, `total`) boosts `table`/`visual` kinds a little. A summary-seeking question (`what does … say about`, `main risks`) turns on parent expansion and a larger N.

**b. Dense search:** `gemini-embedding-001` with `output_dimensionality=768` (MRL-truncated, **re-normalized**), with task types kept (`RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`). Vectors live in a float32 `.npy` matrix, so the scan is a single matmul. 768 dims cuts the store about 4× compared with 3072 (GEP JSON at 3072 dims would be ~50 MB). The dimension is validated in eval, and we fall back to 1536 if recall drops.

**c. Sparse search:** an in-house BM25 (about 60 lines, NumPy, k1=1.5, b=0.75) over `header + text`. The tokenizer keeps numbers, `%`, decimals, `CO2e`, `FY`, and ID-like tokens intact. This catches exact tokens embeddings blur: `FDsys`, `ENERGY STAR`, `802,834`, `Table 2.10`.

**d. Fusion:** RRF `Σ 1/(60 + rank)` across both lists, so no score calibration is needed. **Diversity:** drop `dup_of` siblings and cap chunks per page at 3.

**e. Reranking:** a single Gemini call with `generate_json` scores the top 20 candidates 0–3 for "directly answers / partially / related / irrelevant" against the question, using header + first ~600 chars of each. Keep chunks with score ≥ 2, at least 3 and at most 8. Toggle with `RERANK=llm|off`. This costs one extra call per question. Cross-encoder rerankers (`bge-reranker`) were considered and rejected because they add torch and a ~1 GB model to a lightweight project. The `llm.py` interface leaves room to swap one in later.

**f. Relevance gate:** replaces the raw `cosine < 0.60` rule, which won't transfer to 768-dim or hybrid scores. If the reranker returns no chunk ≥ 2 → "couldn't find it" fallback. With `RERANK=off`, keep a cosine gate whose threshold is calibrated in eval. The **exact-ID guardrail is kept**, but it no longer fires on `Chapter`/`Table`/`Box` references (they are handled by section scope).

### 3.8 Context assembly

- **Parent expansion:** for summary intent, or when ≥ 2 hits share a parent, replace the children with the parent window, deduplicated.
- **Ordering:** group by document → section → reading `order`, so the model reads a coherent narrative instead of a score-ordered jumble.
- **Budget:** about 12k chars of context (`CONTEXT_CHAR_BUDGET`). Drop the lowest-ranked groups first, never in the middle of a table.
- **SOURCE block format:**

  ```text
  [SOURCE 3] GEP-Jun-2026.pdf · Chapter 2 › South Asia › Outlook · p.74 (PDF p.96) · text
  …chunk text…
  Notes: …linked footnotes…
  ```

### 3.9 Generation and grounding (`grounding.py`)

**Prompt changes** to `GROUNDED_PROMPT`:

- "Answer from the sources only. Give exact figures with their units and the year/period/configuration they apply to."
- "If sources cover different regions, chapters, product configurations or years, only use the ones the question asks about. Name which one you used."
- "If sources disagree (e.g. a table and a footnote), report both and say they differ."
- "When a figure comes from a chart transcription, say so."
- Domain-neutral wording replaces the "technical documentation assistant" framing (the UI copy is updated to match).
- Citation behaviour is unchanged from Phase 1 (`[SOURCE n]` markers plus the sources panel).

**Numeric faithfulness check** (no extra LLM call in the common case):

1. Extract numeric claims from the answer (`166`, `$375 million`, `6.3 percent`, `346 kg`, `58%`). Normalize `percent`↔`%`, commas, `million`/`M`, and en-dashes.
2. Each one must appear in the cited context (tolerating rounding of displayed decimals).
3. If some are unsupported, regenerate once with a stricter instruction listing the unsupported numbers. If they're still unsupported, return the answer with those figures flagged, and log it. `GROUNDING_CHECK=on|off`.
4. The response gets `grounding: {"checked": n, "unsupported": [...]}` for the UI and eval.

### 3.10 Store, ingestion and API

**Store (`store.py`):**

- `data/index/meta.jsonl` (chunk records without vectors), `data/index/vectors.npy`, and `data/index/docs.json` (doc cards, profiles, glossary, section trees, file hash). BM25 is rebuilt in memory at load time (milliseconds).
- Atomic writes: temp file then `os.replace`, so a crash mid-ingest can't corrupt the index.
- `INDEX_VERSION = 2`. If an old `index.json` is found, it is ignored with a clear "re-upload your documents" message in `/api/stats` (no silent mixing of v1 and v2 chunks).
- Re-upload replacement keys on `doc_id` (content hash) **and** file name.

**Ingestion jobs (`jobs.py`):** GEP needs about 1,600 embeddings plus up to 30 vision calls, which takes minutes. A synchronous upload would time out or freeze the UI.

- `POST /api/upload` validates, stores the bytes, and returns `{job_id}` per file.
- A worker thread runs parse → structure → tables/footnotes → vision → chunk → embed → store, updating `{stage, done, total, warnings}`.
- `GET /api/jobs/{id}` is polled by the UI. Per-stage progress: "Vision 12/30", "Embedding 640/1600".
- `llm.py` retries `429`/`5xx` with exponential backoff and jitter. Embedding batches stay at 32. Vision calls run with small concurrency (`VISION_CONCURRENCY=3`).
- One job runs at a time (queue), which is enough for a single-user app.

**API changes:**

| Method | Endpoint | Change |
|---|---|---|
| `POST` | `/api/upload` | Returns jobs instead of blocking |
| `GET` | `/api/jobs/{id}` | **New**: progress |
| `GET` | `/api/documents` | **New**: per doc: name, pages, profile, chunk counts by kind, sections (top two levels) |
| `DELETE` | `/api/documents/{doc_id}` | **New**: remove one document |
| `POST` | `/api/ask` | Accepts optional `doc_ids` and `debug`. Returns `sources` with `section_path`, `page_label`, `kind`, and `grounding`. `debug=true` adds query analysis, candidate lists and rerank scores (used by eval) |
| `GET` | `/api/stats` | Adds chunk counts by kind and an index-version warning |

### 3.11 Frontend (`index.html`, still a single file with no build step)

- Upload progress driven by job polling, with stage labels.
- Document list with a profile badge (`narrative`, `long-structured`, `visual-dense`), per-doc delete, and a **scope selector** ("All documents" / one document).
- The citation panel shows `section › path · p.74 (PDF p.96)` and a kind badge (`table`, `chart`, `footnote`).
- A grounding warning chip when unsupported figures were flagged.
- Copy changes from "Technical Docs" to a neutral "Document Q&A" framing.
- Existing responsive and theme behaviour, plus the 360–1920px layout check, are kept.

---

## 4. Evaluation (`backend/eval/`)

### 4.1 Question set (`questions.json`)

Each entry: `id, doc, question, must_contain[] (regex), must_not_contain[] (regex), expected_pages[], expected_section (optional), type (fact | numeric | summary | table | chart | footnote | negative | trap)`.

- **The 9 sample questions** (§0 key).
- **Held-out positives** (about 4 per doc), written to exercise each mechanism. For example: GPO *copies distributed* (802,834 copies of 4,211 titles), *FY 2018 revenues* ($874,466,000). GEP *global growth 2026* (2.5%), *India FY2026/27* (6.6%), *SSA growth 2025* (4.1%), *Maldives 2026f* from Table 2.10 (0.7%), *Box 1.1 on AI*. iMac *sleep power at 115V* (0.57W), *four-port footprint* (391 kg), *packaging recycled content* (76%), *baseline emissions before reductions* (445 kg, which tests the chart), *what power adapter the ENERGY STAR test used* (143W, which tests the footnote).
- **Traps, where wrong-section answers would look plausible:** *"Growth outlook for East Asia and Pacific?"* `must_not_contain: 6\.3 percent` (the SAR figure). *"iMac footprint for the 512GB four-port?"* `must_not_contain: 346`. *"Which Chapter 4 topic…"* must not cite Ch.3 pages.
- **Negatives (must refuse):** *"How many employees did GPO have in FY 2022?"*, *"What is the iMac's battery capacity?"*, *"What does GEP forecast for 2030?"*. These pass only on a "not found" answer containing no fabricated numbers.

### 4.2 Metrics (`run_eval.py`)

| Metric | Definition |
|---|---|
| **Retrieval hit@k** | Any expected page is in the final context |
| **MRR** | Rank of the first expected-page chunk after rerank |
| **Section precision** | Share of context chunks from the expected section (Use Case 2 bar: "did not mix up regions") |
| **Answer pass** | All `must_contain` match **and** no `must_not_contain` match |
| **Refusal accuracy** | Negatives correctly refused, positives not refused |
| **Unsupported numbers** | From the grounding check |
| Latency / LLM calls per question | Cost visibility |

Optional: an LLM-as-judge faithfulness score (`--judge`) for summary questions W2/W3, where regexes are too blunt. Report it, but don't gate on it.

### 4.3 Ablation

`run_eval.py --config` flips each stage independently: `PARSER=pypdf|pymupdf`, `CHUNKING=fixed|structural`, `CONTEXT_HEADERS`, `TABLES`, `FOOTNOTES`, `VISION`, `HYBRID`, `RERANK`, `GROUNDING_CHECK`. Output is a markdown table per run written to `eval/results/`, with a final committed `plans/phase 2/EVAL-RESULTS.md` showing **baseline → each stage → final**. That table is the deliverable that answers *"why does this approach work better for this document type"*.

### 4.4 Definition of done

- **9/9 sample questions pass**, judged by the regex key plus a manual read of W2/W3.
- Held-out positives ≥ 90% pass. **All traps pass. All negatives refuse.**
- GEP section precision ≥ 0.8 on region/chapter questions.
- 0 unsupported numbers in final answers across the set.
- Offline unit suite green without an API key. The existing 39 Phase 1 assertions still pass, or are updated where behaviour intentionally changed.

---

## 5. Milestones

Each milestone ends with an eval run, and is only merged if it moves its target metric without regressing others.

| # | Milestone | Main work | Target metric / exit criterion |
|---|---|---|---|
| **M0** | **Eval harness + baseline** | `questions.json`, `run_eval.py`, `debug` on `/api/ask`. Index the 3 PDFs with the **unchanged** Phase 1 pipeline | A baseline table exists. Expected: GPO mostly passes. W1 (SAR acronym), W2/W3 (section), A2 (config ambiguity), table/chart held-outs and traps fail. *This is the "before" picture* |
| **M1** | Parsing layer | PyMuPDF, text repair, header/footer harvest + removal, page labels, front-matter skip, glossary, block classification, reading order. Unit tests on the real PDFs | No `Te `/`7T_h`/`�` artefacts. TOC pages excluded. `page_label` correct on sampled pages. GPO holds or improves |
| **M2** | Structure + structural chunking + contextual headers | Section tree (outline + reconciliation + headings), profiles, parent/child chunks, headers, near-dup marking. **New store format** (npy/jsonl, 768-dim, versioning) | W1 passes. Section precision on GEP rises sharply. South Asia section starts on p96 in tests |
| **M3** | Tables + footnotes | Coordinate table rebuild, dual serialization, superscript markers, endnote parsing, linking, footnote chunks | A1 cites the footnote. A2 states 346/50/45 with configuration. Table held-outs (Maldives 2026f, 0.57W) pass |
| **M4** | Vision enrichment | Page scoring, rendering, Gemini JSON transcription, cache, budget, trust labelling | iMac p2 chart held-out (445 kg baseline, −99 kg) passes. Vision cache makes re-index free |
| **M5** | Hybrid retrieval + query analysis + rerank | BM25, RRF, diversity, doc/section scoping, doc cards, acronym expansion, LLM rerank, new relevance gate | W2/W3 pass with correct section. All traps pass. Negatives refuse. Retrieval hit@k ≥ 0.95 |
| **M6** | Context assembly + grounding | Parent expansion, ordering, budget, prompt rules, numeric faithfulness check + one retry | 0 unsupported numbers. W2/W3 summaries complete (judge spot-check) |
| **M7** | Ingestion jobs, API, UI | Background jobs + progress, `/api/documents`, scope selector, citation panel upgrades, grounding chip | GEP uploads through the UI with live progress and no request timeout. Layout check passes at 360–1920px |
| **M8** | Hardening + write-up | Full ablation run, `EVAL-RESULTS.md`, README update (per-document strategy rationale, new config, limits) | Definition of done (§4.4) met |

M1–M4 are mostly offline-testable. M0, M4, M5 and M8 need a live API key.

---

## 6. Per-document strategy summary (what the write-up will argue)

| | GPO (narrative) | GEP (long, structured) | iMac (visual, dense) |
|---|---|---|---|
| Main failure in Phase 1 (to be confirmed by M0) | Near-duplicates and header noise; mostly fine | Wrong region/chapter (acronyms, seven similar "Outlook" sections), TOC pages out-scoring content, fragments for summary questions | Chart unreadable, footnotes detached, two configurations blended, table glued to notes |
| Decisive techniques | Clean parsing, paragraph-packed chunks, hybrid for exact numbers, duplicate diversity | Section tree + contextual headers, glossary, section scoping, parent expansion, rerank | Table rebuild, footnote linking, vision transcription, configuration-aware prompt, numeric grounding |
| Expected to matter least | Vision, section scoping | Vision (captions already state each figure's message) | Section tree (12 pages) |

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Vision invents chart values | Strict "transcribe only" JSON schema, cross-check against text-layer labels, trust label in context, grounding check, cached outputs reviewed once by hand for iMac p2 |
| Configured chat model rejects image input in the user's region | `VISION_MODEL` is its own setting. Vision degrades to `off` with a warning, and the rest of the pipeline is unaffected |
| API rate limits during a 200-page ingest | Batching, backoff with jitter, bounded concurrency, vision cache, resumable jobs (re-running skips cached vision pages) |
| Heuristics (headers, tables, headings) tuned to three PDFs and brittle elsewhere | Thresholds are relative (body font size, % of pages), every heuristic has a safe fallback to plain text, and tests include the Phase 1 sample PDFs as a non-regression set |
| Reranker cost/latency | One call per question, top-20 inputs truncated to ~600 chars, `RERANK=off` toggle, latency reported in eval |
| 768-dim embeddings lose recall | Measured in M2. `EMBEDDING_DIM` setting, fall back to 1536/3072 |
| Over-fitting to the 9 sample questions | Held-out and trap questions written before M1 and not edited to make a milestone pass |
| PyMuPDF AGPL licence | Documented (§3.1). The pdfplumber + pypdfium2 swap path is noted |

---

## 8. Out of scope for Phase 2

- OCR for scanned PDFs (none of the three needs it, since all have a text layer).
- Multi-user or tenant isolation, auth, and an external vector database (Phase 1 limits table still applies).
- Streaming responses, and fine-tuning or training a custom embedding model.
- Non-PDF formats.

---

## 9. New configuration (`backend/.env`)

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_DIM` | `768` | MRL output dimensionality (re-normalized) |
| `CHUNK_TARGET` / `CHUNK_MAX` | `900` / `1400` | Child chunk size in chars (replaces `CHUNK_SIZE`/`CHUNK_OVERLAP`) |
| `PARENT_MAX` | `3500` | Parent window size |
| `CANDIDATES` | `40` | Dense and BM25 candidates each, before fusion |
| `RERANK` | `llm` | `llm` or `off` |
| `RERANK_TOP` | `20` | Candidates sent to the reranker |
| `CONTEXT_CHAR_BUDGET` | `12000` | Context size cap |
| `VISION_MODE` | `auto` | `off`, `auto` or `all` |
| `VISION_MODEL` | `= CHAT_MODEL` | Multimodal model for page transcription |
| `VISION_MAX_PAGES` | `30` | Per-document vision budget |
| `VISION_CONCURRENCY` | `3` | Parallel vision calls |
| `GROUNDING_CHECK` | `on` | Numeric faithfulness check |
| `RELEVANCE_THRESHOLD` | recalibrated in M5 | Only used when `RERANK=off` |
| `MAX_FILE_SIZE_MB` | `20` → `30` | Headroom (GEP is 4.7 MB, iMac 6.8 MB, so this isn't strictly needed) |

New dependency: `pymupdf`. Nothing else heavy: BM25, RRF, table rebuild and grounding are in-house NumPy/stdlib code.
