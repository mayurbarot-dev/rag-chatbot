# How to Test Phase 2

There are three ways to test Phase 2, from quickest to most thorough:

1. **[Manual UI test](#1-manual-ui-test-15-minutes)** (~15 min): upload the three PDFs and ask the checklist questions.
2. **[Automated offline tests](#2-automated-offline-tests-1-minute)** (~1 min, no API key): parsing, tables, footnotes, structure, routing.
3. **[Live evaluation](#3-live-evaluation-1530-minutes)** (15–30 min, uses the Gemini API): 34 graded questions, the Phase 1 baseline and ablations.

All commands run from the repository root in a Windows terminal unless they say otherwise.

---

## 0. One-time setup

1. Make sure `backend/.env` exists with your key. Start from [backend/.env.example](../../backend/.env.example):

   ```text
   GOOGLE_API_KEY=your_key_here
   EMBEDDING_PROVIDER=local
   ```

   `EMBEDDING_PROVIDER=local` is the default and is what the eval was verified with. It avoids the free Gemini tier's embedding quota.

2. Install dependencies. `START_PROJECT.bat` does this for you; manually:

   ```bash
   cd backend
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. **If you used Phase 1 before:** its `backend/data/index.json` can't be read by Phase 2. The app shows a "re-upload your documents" notice. Delete the file or just upload again.

The first upload downloads the local embedding model (~200 MB) once, into `backend/data/cache/models`.

---

## 1. Manual UI test (~15 minutes)

1. Double-click **`START_PROJECT.bat`**, or run:

   ```bash
   cd backend
   .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

   Then open <http://127.0.0.1:8000>.

2. **Check the server is healthy.** The bottom-left corner of the sidebar should show a green dot and **Connected**.

3. **Upload** the three files from [`plans/phase 2/`](.) together, by drag-and-drop or by clicking the drop zone:
   - `GPO-Anual-report.pdf`
   - `GEP-Jun-2026.pdf`
   - `iMac_PER_Oct2024.pdf`

   **What you should see:** a progress toast per file that moves through *Reading PDF → Rebuilding tables and structure → Transcribing charts n/N → Embedding n/N → Saving index*, then "N chunks indexed".

   **Expected time:** iMac under a minute, GPO 1–3 minutes, GEP 5–10 minutes. The first run transcribes up to 30 chart pages and embeds ~830 chunks. Re-uploads are near-instant because the caches are reused.

4. **Check the knowledge base panel.** You should see roughly:

   | File | Badge | Pages | Chunks (approx.) | Extras |
   |---|---|---|---|---|
   | GPO-Anual-report.pdf | NARRATIVE | 82 | ~220 | 17 tables, a few charts |
   | GEP-Jun-2026.pdf | LONG-STRUCTURED | 200 | ~830 | ~48 tables, ~110 charts, ~37 footnotes |
   | iMac_PER_Oct2024.pdf | VISUAL-DENSE | 12 | ~68 | 1 table, ~9 charts, 19 footnotes |

   Chart counts can vary slightly between runs because they depend on the vision model's output.

5. **Ask the checklist questions** below. For each one, open the **sources** panel under the answer. It is collapsed by default; click "N sources".

### 1.1 Use Case 1: GPO Annual Report (narrative)

| # | Ask | Expected answer | Expected source |
|---|---|---|---|
| G1 | How many depository libraries did GPO staff visit in FY 2018? | **166** (in 25 states) | p.2–3 or p.19 (PDF p.4–5 / p.21) |
| G2 | What new digital system replaced FDsys, and when was it fully deployed? | **govinfo**, with full-scale deployment over **FY 2018** (FDsys retirement planned for December 2018) | PDF p.4, p.8 |
| G3 | How much did GPO procure in printing and information products in FY 2018 on behalf of federal agencies? | **More than $375 million** | PDF p.5 (or p.14, $375.7 million) |

### 1.2 Use Case 2: World Bank GEP (long, structured)

| # | Ask | Expected answer | Expected source |
|---|---|---|---|
| W1 | What is the growth outlook for South Asia in this report? | Growth softens to **6.3% in 2026**, then about **7% a year in 2027–28** | `Chapter 2 › South Asia › Outlook`, p.74 (PDF p.96) and Tables 2.9/2.10 (PDF p.98) |
| W2 | What does Chapter 3 say about sovereign debt in developing economies? | Rising debt raises sovereign spreads and yields. The effect is non-linear and larger for vulnerable countries. Debt went from under 40% to over 70% of GDP since 2010 | Every source under `Chapter 3. A Rising Challenge…` |
| W3 | According to the executive summary, what are the main risks to the global outlook? | Escalating hostilities and commodity disruptions, inflation, financial stress (growth could fall to 1.3%), trade uncertainty, geopolitical strains, weather shocks. Upside: AI | `Executive Summary`, p.xvii (PDF p.19) **only** |
| Trap | What is the growth outlook for East Asia and Pacific in 2026? | **4.2%**, and it must **not** say 6.3% (South Asia's figure) | `East Asia and Pacific › Outlook` |

### 1.3 Use Case 3: iMac Environmental Report (visual)

| # | Ask | Expected answer | Expected source |
|---|---|---|---|
| A1 | How much less energy does iMac use compared to the ENERGY STAR requirement? | **58% less** | `Use`, p.7. The source text ends with *Notes: [18] … 143W Power Adapter …* |
| A2 | What is the total carbon footprint of the iMac, and how much of that comes from product use versus manufacturing? | **346 kg CO2e** (two ports, 256GB): **production 50%, product use 45%** (transport 5%) | `Carbon Footprint`, p.10 |
| A3 | What material is the iMac stand made from, and what percentage of it is recycled? | **Aluminum, 100% recycled** | p.1, p.2 or p.4 |
| Table | What is the iMac's power consumption in sleep mode at 115V? | **0.57W** | p.12, kind **table** |
| Chart | What were the iMac's baseline emissions before reductions, and how large were the emissions reductions? | **445 kg** baseline, **−99 kg** reductions, "from the chart" | p.2, kind **chart** |
| Footnote | Which power adapter was the iMac tested with for its ENERGY STAR energy figures? | **143W** Power Adapter | p.12 endnote 18 |

### 1.4 Refusals and guardrails

| Ask | Expected |
|---|---|
| What does the report say about the economic impact of the FIFA World Cup? | "I couldn't find anything in the uploaded documents…", with **no sources** |
| What is the battery capacity of the iMac? | Refusal (an iMac has no battery) |
| How many employees did GPO have in FY 2022? | Refusal (the report covers FY 2018) |
| `Hi` | A friendly reply with no sources (no retrieval) |
| `My name is Dev`, then `What is my name?` | "Your name is Dev." (memory, not documents) |

### 1.5 UI features to try

- **Scope selector** (top bar, appears with 2+ documents): select `iMac_PER_Oct2024.pdf` and ask about South Asia. You should get a refusal, because only the iMac report is searched.
- **Remove a document:** click the ✕ on a document row and confirm. Counts update, and it disappears from the scope list.
- **Grounding chip:** if an answer ever contains a figure that isn't in its sources, a red "⚠ Check: … not found in the sources" chip appears under it. This is rare.
- **Theme toggle** (sun/moon, top left) and phone layout (narrow the window below 860px; the sidebar becomes a drawer).

### 1.6 What a pass looks like

- Each answer has the right figure **and** its sources come from the expected section and page.
- Answers name the region, configuration or period they use (e.g. "iMac (two ports) with 256GB").
- Out-of-scope questions are refused instead of answered from a nearby passage.

---

## 2. Automated offline tests (~1 minute)

No API key or network needed: the model is replaced by a deterministic fake.

```bash
backend\.venv\Scripts\python.exe backend\tests\test_documents.py
backend\.venv\Scripts\python.exe backend\tests\test_pipeline.py
```

Expected output ends with:

```text
  67 passed, 0 failed      (test_documents.py)
  51 passed, 0 failed      (test_pipeline.py)
```

| Suite | What it covers |
|---|---|
| `test_documents.py` | On the real Phase 2 PDFs: "Th" ligature repair, footer removal, drop caps, printed page labels (`p.74`, `xvii`, `c1`), contents and abbreviations pages skipped, glossary (SAR = South Asia), South Asia snapped from outline p95 to its heading on p96 under Chapter 2, Box 1.1 extent, Executive Summary section, Table 2.10 (Maldives 2026f = 0.7), GPO balance sheet columns, iMac power table (sleep at 115V = 0.57W), 19 endnotes with note 18 linked to the p7 claim, key/value joins (`GPO staff: 1,737`), numeric grounding rules, query analysis (document routing, Chapter/Table/Executive Summary filters, section boost, acronym expansion) |
| `test_pipeline.py` | The full pipeline on the Phase 1 sample PDFs: indexing, reload after restart, re-upload replacement, bad-PDF rejection, every routing branch (memory, name, small talk, empty index, grounded, reranker refusal, cosine fallback gate, exact-ID guardrail), the grounding retry, memory/document separation |

A failure prints `FAIL <check name>` with the actual value. Each check name says which behaviour broke.

---

## 3. Live evaluation (15–30 minutes)

The live eval uses the Gemini API for answers, reranking and chart transcription. It indexes the three PDFs into its own directory, `backend/eval/.work`, so **your app's index is never touched**.

```bash
cd backend
set EMBEDDING_PROVIDER=local

:: Phase 2, all stages on (the first run indexes everything; later runs reuse the index)
.venv\Scripts\python.exe eval\run_eval.py --label my-run

:: Only some questions, printing every answer
.venv\Scripts\python.exe eval\run_eval.py --only W1,W3,A2 --verbose

:: The frozen Phase 1 engine, as a baseline (separate work dir)
.venv\Scripts\python.exe eval\run_eval.py --pipeline phase1 --label baseline --work-dir eval\.work-baseline

:: Force re-indexing after changing parsing or chunking code
.venv\Scripts\python.exe eval\run_eval.py --reindex --label after-change
```

**Ablation.** Turn one stage off with `--set KEY=VALUE`. Index-time settings need their own `--work-dir`:

```bash
.venv\Scripts\python.exe eval\run_eval.py --label no-rerank --set RERANK=off
.venv\Scripts\python.exe eval\run_eval.py --label no-vision --set VISION_MODE=off --work-dir eval\.work-novision
```

Or run the whole set (Git Bash): `bash eval/run_ablations.sh`.

### 3.1 Reading the report

Each run writes `backend/eval/results/<timestamp>-<label>.md` (a summary plus per-question table) and `.json` (full answers and sources).

| Column | Meaning |
|---|---|
| Pass | Answer has every required fact and no forbidden one; refusal questions pass when refused |
| Strict pass | Pass **and** an expected page was actually in the context. A right answer from the wrong section fails this |
| Hit@k | Share of questions where an expected page was retrieved |
| MRR | How high the first correct source ranks (1.0 = always first) |
| Section precision | Share of sources from the expected section (region and chapter questions) |
| Unsupported numbers | Figures in answers that don't appear in their sources |

**Reference numbers** are in [EVAL-RESULTS.md](EVAL-RESULTS.md). Small run-to-run differences come from the hosted model (answers use temperature 0.2). See EVAL-RESULTS.md for reference runs.

> **Quota:** a full run makes ~80 Gemini generate calls (plus ~40 chart transcriptions on the first index). The free tier's daily cap allows only a few full runs per day. When it runs out, every remaining question shows `ERROR: … daily quota is exhausted` and the run is invalid. Re-run the next day.

### 3.2 Adding your own questions

Add an entry to [backend/eval/questions.json](../../backend/eval/questions.json):

```json
{"id": "W12", "doc": "gep", "type": "numeric",
 "question": "What is Europe and Central Asia's growth forecast for 2026?",
 "must_contain": ["2\\.1 ?(percent|%)"], "expected_pages": [84, 85, 86]}
```

`type` is one of `sample`, `numeric`, `table`, `chart`, `footnote`, `summary`, `trap` or `negative`. For `negative`, leave `must_contain` empty: a refusal is the pass. Pages are **PDF** page numbers, not printed labels.

---

## 4. Checking the API directly (optional)

With the server running:

```bash
curl http://127.0.0.1:8000/api/health
curl -F "files=@plans/phase 2/iMac_PER_Oct2024.pdf" "http://127.0.0.1:8000/api/upload?wait=true"
curl http://127.0.0.1:8000/api/documents
curl -X POST -H "Content-Type: application/json" ^
     -d "{\"question\": \"What is the iMac's power consumption in sleep mode at 115V?\", \"debug\": true}" ^
     http://127.0.0.1:8000/api/ask
```

With `"debug": true`, the response includes `debug.plan` (document scope, filters, boosted sections, acronym expansion), `debug.lists` (top dense and BM25 candidates) and `debug.rerank` (each candidate's 0–3 score). This is the quickest way to see *why* a question got the sources it did.

Interactive API docs are at <http://127.0.0.1:8000/docs>.

---

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "No API key" in the sidebar | Add `GOOGLE_API_KEY` to `backend/.env` and restart |
| Upload toast stuck on *Transcribing charts* | The Gemini free tier allows ~14 calls/minute; 30 GEP pages take 2–3 minutes. `retrying` lines in the server window are normal |
| "daily quota is exhausted" | The free-tier daily limit is used up. Wait until tomorrow, use a paid key, or make sure `EMBEDDING_PROVIDER=local` |
| "built with a different embedding model" | You changed `EMBEDDING_PROVIDER`. Click **Clear documents** and upload again |
| Old Phase 1 documents missing | The Phase 1 index format isn't compatible. Upload again |
| A chart question isn't answered | Check that the page was transcribed: `/api/documents` lists `vision_pages`. Raise `VISION_MAX_PAGES` or set `VISION_MODE=all` and re-upload |
| An answer cites the wrong region or chapter | Ask again with `"debug": true` and look at `debug.plan` and `debug.rerank`; [TECHNICAL-GUIDE.md](../../TECHNICAL-GUIDE.md) §13–15 explains each step |
