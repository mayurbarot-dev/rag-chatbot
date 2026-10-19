# Phase 2 Evaluation Results

**Dates:** 2026-10-01 (baseline, 28-question run, ablations) and 2026-10-13 (34-question run after multi-part handling) · **Question set:** [backend/eval/questions.json](../../backend/eval/questions.json). It has 28 questions: the 9 sample questions, 14 held-out questions, 2 traps and 3 refusals. It was written before any pipeline work and not edited afterwards.

**Setup, identical across runs:**
- Answers, reranking and chart transcription use `gemini-3.1-flash-lite`.
- Embeddings use the local `BAAI/bge-base-en-v1.5`, so the comparison isolates the *pipeline*; the embedding model is held constant.
- The baseline is the frozen Phase 1 engine ([eval/phase1_rag.py](../../backend/eval/phase1_rag.py)). It uses pypdf, 1000-character fixed chunks, cosine top-5 and a 0.60 relevance gate.

How to reproduce: [HOW-TO-TEST.md §3](HOW-TO-TEST.md#3-live-evaluation-1530-minutes).

---

## 1. Headline

**Latest run (2026-10-13, 34 questions; 6 held-out questions added before the run):** **34/34 pass, 34/34 strict**, hit@k 1.00, MRR 0.90, section precision 1.00. G2 now answers both parts, after the multi-part handling (question split into sub-searches plus a checklist in the prompt). One figure was flagged as unsupported: "2028", which the table writes as "2028f". That was a checker false positive, now fixed.

The comparison below is from the 28-question run, which the baseline and ablations share.

| | Phase 1 baseline | **Phase 2** |
|---|---|---|
| Pass (correct answer) | 23/28 | **28/28** |
| Strict pass (correct answer **and** a correct page retrieved) | 22/28 | **28/28** |
| Sample questions (the 9 in the brief) | 7/9 | **9/9** |
| Hit@k (an expected page is in the context) | 0.88 | **1.00** |
| MRR (rank of the first correct source) | 0.73 | **0.91** |
| Section precision (sources from the asked-about section) | 0.53 | **1.00** |
| Unsupported numbers in answers | 0 | **0** |
| Average latency per question | 4.3 s | 9.6 s |

Phase 2 is about twice as slow per question. Each answer adds a rerank call, and the free tier's pacing is applied to every call.

### Per document

| Document | Baseline pass | Phase 2 pass | Baseline MRR | Phase 2 MRR |
|---|---|---|---|---|
| GPO (narrative) | 5/7 | **7/7** | 0.53 | **0.92** |
| GEP (long, structured) | 9/11 | **11/11** | 0.75 | **1.00** |
| iMac (visual) | 9/10 | **10/10** | 0.83 | 0.80 |

### What the baseline got wrong

| Id | Question | Baseline failure | Phase 2 fix |
|---|---|---|---|
| W1 | Growth outlook for South Asia | Missed "6.3%": the section only ever says "SAR" | Glossary alias in the chunk header (`South Asia (SAR)`), section boost, BM25 `sar` expansion |
| W3 | Exec summary: main risks | Passed the regex, but **no Executive Summary page was retrieved**; it answered from Chapter 1 | "executive summary" becomes a hard section filter → only p.xvii–xviii |
| W4 | Global growth in 2026 | No expected page retrieved | Section-aware chunks + rerank |
| G2 | What replaced FDsys, when deployed | Omitted the deployment period | Passes intermittently, see §4 |
| G6 | Passports produced in FY 2018 | The "By the Numbers" value was split from its label | Key/value merge: `Passports produced: 15,901,600` |
| A9 | Share of manufacturing electricity from renewables | Not retrieved | Cleaner parsing; icon call-outs transcribed by vision |

---

## 2. Ablation: what each stage contributes

Each row switches **one** stage off and re-runs all 28 questions.

| Configuration | Pass | Hit@k | MRR | Section precision | Unsupported | Questions lost |
|---|---|---|---|---|---|---|
| **Full Phase 2** | **28/28** | 1.00 | 0.91 | 1.00 | 0 | — |
| `RERANK=off` (cosine gate) | 24/28 | 1.00 | 0.84 | 1.00 | 2 | G2, **all 3 refusals** |
| `HYBRID=off` (dense only) | 26/28 | 0.96 | 0.85 | 1.00 | 0 | G2, **W8 trap** (wrong region) |
| `QUERY_SCOPING=off` | 27/28 | 1.00 | 0.93 | **0.83** | 0 | G2 |
| `VISION_MODE=off` | 27/28 | 1.00 | 0.91 | 1.00 | 0 | **A7 chart** |
| `TABLES=off` | 26/28 | 0.96 | 0.85 | 1.00 | 0 | G2, **A4 table** |
| `CONTEXT_HEADERS=off` | 27/28 | 1.00 | 0.89 | 1.00 | 0 | G2 |

How to read it:

- **The reranker is the refusal mechanism.** Without it, all three out-of-scope questions get answered from the nearest passage, and two answers contain figures not in their sources. Similarity search always returns *something*; only a relevance judgement can say "none of this answers it".
- **Hybrid search protects the trap question.** Dense-only retrieval let South Asia passages displace East Asia & Pacific ones for W8. Keyword matches on the exact region name pin the right section.
- **Query scoping keeps sources on-section.** Answers stayed correct without it, but section precision fell from 1.00 to 0.83. One in six sources came from the wrong chapter or region, which is exactly the mixing Use Case 2 forbids, and a stronger prompt can't always ignore it.
- **Vision is the only way to answer A7.** 445 kg baseline and −99 kg reductions exist only in the p2 waterfall chart.
- **Table reconstruction is the only way to answer A4** (0.57W, sleep at 115V). Flattened text loses which value belongs to which voltage column.
- **Contextual headers** mostly improve ranking (MRR 0.91 → 0.89 without them). At this question count their effect on pass rate is within noise.

G2 appears in most "lost" lists because it's unstable, not because of those stages (§4).

---

## 3. Mapping to the brief's three use cases

| Use case | Pass bar from the brief | Result | Mechanisms that mattered (measured) |
|---|---|---|---|
| 1. Plain-text report (GPO) | Exact fact from the surrounding paragraph, nothing added | 3/3 sample, 7/7 overall; 0 unsupported numbers | Clean parsing (apostrophes, drop caps, footer removal), key/value merge, near-duplicate diversity, numeric grounding check |
| 2. Long structured report (GEP) | Right chapter and section; no region or chapter mix-ups | 3/3 sample, 11/11 overall, **section precision 1.00**, trap passed | Outline snapped to real headings, section paths and aliases in chunk headers, hard section filters, named-section boost, hybrid search, reranker's wrong-region rule, parent windows for summaries |
| 3. Visual document (iMac) | The right number from a chart or table, not a nearby sentence | 3/3 sample, 10/10 overall (chart, table, footnote, configuration trap) | Vision transcription, coordinate table rebuild, footnote linking, configuration-aware prompt rules |

---

## 4. Known issues and limits (stated plainly)

1. **G2 was unstable; fixed 2026-10-13.** Before the fix: In 8 runs of Phase 2 code (full + 6 ablations + 1 partial), G2 passed **2 times**. The answer always names govinfo and cites the right pages. It usually reports "out of beta December 2017" and "FDsys retirement December 2018", but omits the sentence "full-scale deployment … over the course of FY 2018". That sentence is retrieved but not used. Adding a "answer every part of the question" rule and temperature 0 did **not** fix it in the one run that completed before the quota ran out (§5), so that change was reverted. Query decomposition (`split_parts` in retrieval.py) fixed it: each part is searched separately and listed as a checklist in the prompt. It passed in the 34-question run.
2. **The iMac footnote contradiction is not flagged.** The p10 table attributes 346 kg to "two ports, 256GB", while endnote 10 says the calculation is based on "two ports with 512GB". A2's answer reports the table correctly but doesn't mention the conflict. The prompt asks for disagreements to be reported, but endnote 10 is attached to the p2 chart labels, not to the p10 chunk the answer used.
3. **iMac MRR (0.80) trails the baseline (0.83).** Answers are correct, but for A2, A7 and A8 the best source is ranked 2nd or 3rd behind related passages. The rerank keeps it and the answer uses it, so pass rate is unaffected.
4. **Small question set.** 28–34 questions can show large effects (refusals, charts, tables, wrong region), but differences of one question are within run-to-run noise from the hosted model.
5. **Embeddings.** All numbers use the local embedding model. The Gemini embedding path (`EMBEDDING_PROVIDER=gemini`) is implemented and unit-tested at the API level (768-dim vectors returned). It was **not** run through the full eval, because indexing the three reports (~1,100 chunks) exceeds the free tier's daily embedding allowance.
6. **Heuristics are validated on 5 PDFs** (the 3 Phase 2 reports and the 2 Phase 1 samples). Tables, headings and footnote rules use relative signals with safe fallbacks, but an unfamiliar layout may need tuning. `tests/test_documents.py` is the regression net.

---

## 5. Run log

| Run | Result file (in `backend/eval/results/`, not committed) | Notes |
|---|---|---|
| Phase 1 baseline | `20261001-145216-baseline-phase1.*` | 28 questions, local embeddings |
| Phase 2 full | `*-phase2-full.*` | 28/28; the headline numbers |
| 6 ablations | `*-no-rerank.*`, `*-no-hybrid.*`, `*-no-scoping.*`, `*-no-vision.*`, `*-no-tables.*`, `*-no-headers.*` | One stage off each |
| Phase 2 "final" (prompt tweak) | `*-phase2-final.*` | **Invalid.** The Gemini daily generation quota ran out after question W6; every later question errored. The tweak it tested was reverted. |

The free tier allowed roughly 4–5 full runs of 28 questions per day on this key, plus the one-time chart transcription. Plan eval runs accordingly.
