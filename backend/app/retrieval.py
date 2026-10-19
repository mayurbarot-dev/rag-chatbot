"""Query analysis, hybrid retrieval, reranking and context assembly.

question
  -> analyse: which document(s), which section(s), acronyms, intent
  -> dense top-N and BM25 top-N (plus the same within a named section)
  -> reciprocal rank fusion -> near-duplicate / per-page diversity
  -> LLM rerank (the relevance gate) -> context blocks, parents expanded
"""

import re
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from . import config
from .store import Store, tokenize
from .structure import GENERIC_TITLES

RRF_K = 60

SUMMARY_RE = re.compile(
    r"\b(what does .{0,60}\b(say|said|tell|find|conclude)|summar\w*|overview|"
    r"main (risks?|findings?|points?|messages?|drivers?|themes?)|key (findings?|points?|messages?|risks?)|"
    r"describe|explain|discuss)\b",
    re.IGNORECASE,
)
NUMERIC_RE = re.compile(
    r"\b(how (much|many)|what (percentage|share|percent|proportion|amount)|total|"
    r"rate|number of|forecast|projected|estimate)\b",
    re.IGNORECASE,
)
CHAPTER_Q = re.compile(r"\b(?:chapter|ch\.?)\s*(\d+)\b", re.IGNORECASE)
BOX_Q = re.compile(r"\bbox\s+([A-Z]?\d+(?:\.\d+)*)\b", re.IGNORECASE)
TABLE_Q = re.compile(r"\btable\s+([A-Z]?\d+(?:\.\d+)*)\b", re.IGNORECASE)
FIGURE_Q = re.compile(r"\bfigure\s+([A-Z]?\d+(?:\.\d+)*)\b", re.IGNORECASE)
FOOTNOTE_Q = re.compile(r"\b(?:foot|end)note\s+(\d+)\b", re.IGNORECASE)
NAMED_PARTS = ("executive summary", "statistical annex", "statistical appendix", "definitions", "endnotes")


@dataclass
class QueryPlan:
    doc_ids: set[str] | None = None
    doc_reason: str = ""
    hard_filter: object = None  # callable(chunk) -> bool
    hard_reason: str = ""
    soft_sections: list[str] = field(default_factory=list)  # section titles to boost
    bm25_extra: list[str] = field(default_factory=list)
    summary: bool = False
    numeric: bool = False
    parts: list[str] = field(default_factory=list)  # sub-questions of a multi-part question


PART_SPLIT_RE = re.compile(r",?\s+and\s+(?=(?:when|how|what|which|where|why|who|whom|whose)\b)", re.IGNORECASE)


def split_parts(question: str) -> list[str]:
    """'What replaced FDsys, and when was it fully deployed?' -> two sub-questions.

    A multi-part question embeds as a blend of its parts, so the passage that
    answers the second part can rank low, and even when it is retrieved the
    model tends to answer only the first part. Each part is searched on its
    own and listed for the model as a checklist.
    """
    pieces = [p.strip(" ,?") for p in PART_SPLIT_RE.split(question) if p.strip(" ,?")]
    return [p + "?" for p in pieces] if len(pieces) > 1 else []


@dataclass
class Retrieval:
    blocks: list[dict]
    refused: bool
    plan: QueryPlan
    debug: dict


class RerankScore(BaseModel):
    id: int
    score: int


class RerankResult(BaseModel):
    scores: list[RerankScore]


RERANK_PROMPT = """You are ranking passages retrieved for a question about the uploaded documents.

Score every passage for how useful it is to answer the question:
3 = directly contains the specific facts or figures asked for, for the right entity, \
region, chapter, period or product configuration
2 = answers part of the question, or gives context the answer needs
1 = same topic but does not help answer this question
0 = irrelevant

A passage about a different region, chapter, year or product configuration than the \
one the question asks about scores at most 1, even if it looks similar.

QUESTION: {question}

PASSAGES:
{passages}

Return a score for every passage id."""


# --------------------------------------------------------------------------
# Query analysis
# --------------------------------------------------------------------------


def _phrase_in(phrase: str, text: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])", text, re.IGNORECASE) is not None


def analyse(store: Store, question: str, doc_ids: list[str] | None = None) -> QueryPlan:
    plan = QueryPlan(summary=bool(SUMMARY_RE.search(question)), numeric=bool(NUMERIC_RE.search(question)))

    # 1. Document scope: the UI's choice, or terms that only one document uses.
    if doc_ids:
        plan.doc_ids = set(doc_ids)
        plan.doc_reason = "selected in the UI"
    elif config.QUERY_SCOPING and len(store.docs) > 1:
        plan.doc_ids, plan.doc_reason = _route_document(store, question)

    if not config.QUERY_SCOPING:
        return plan

    scoped_docs = plan.doc_ids or set(store.docs)
    sections = {}
    for c in store.chunks:
        if c["doc_id"] in scoped_docs:
            for title in c["section_path"]:
                sections.setdefault(title, set()).add(c["doc_id"])
    glossary: dict[str, str] = {}
    aliases: dict[str, list[str]] = {}
    for d in scoped_docs:
        info = store.docs.get(d, {})
        glossary.update(info.get("glossary", {}))
        for title, al in info.get("aliases", {}).items():
            aliases.setdefault(title, []).extend(al)

    # 2. Explicit structural references are hard filters.
    filters = []
    reasons = []
    m = CHAPTER_Q.search(question)
    if m:
        want = f"chapter {m.group(1)}"
        titles = [t for t in sections if t.lower().startswith(want + ".") or t.lower() == want]
        if titles:
            filters.append(lambda c, t=set(titles): any(p in t for p in c["section_path"]))
            reasons.append(titles[0])
    m = BOX_Q.search(question)
    if m:
        want = f"box {m.group(1).lower()}"
        titles = [t for t in sections if t.lower().startswith(want + " ") or t.lower() == want]
        if titles:
            filters.append(lambda c, t=set(titles): any(p in t for p in c["section_path"]))
            reasons.append(titles[0])
    for part in NAMED_PARTS:
        if part in question.lower():
            titles = [t for t in sections if t.lower() == part]
            if titles:
                filters.append(lambda c, t=set(titles): any(p in t for p in c["section_path"]))
                reasons.append(titles[0])
    for regex, kind, word in ((TABLE_Q, "table", "table"), (FIGURE_Q, "figure", "figure")):
        m = regex.search(question)
        if m:
            label = f"{word} {m.group(1)}".lower()
            pat = re.compile(r"\b" + re.escape(label) + r"\b", re.IGNORECASE)
            if any(c["kind"] in (kind, "visual") and pat.search(c["text"][:200]) for c in store.chunks):
                filters.append(lambda c, p=pat, k=kind: c["kind"] in (k, "visual") and bool(p.search(c["text"][:200])))
                reasons.append(label)
    m = FOOTNOTE_Q.search(question)
    if m:
        num = m.group(1)
        filters.append(lambda c, n=num: c["kind"] == "footnote" and c["text"].startswith(f"Footnote {n}:"))
        reasons.append(f"footnote {num}")
    if filters:
        plan.hard_filter = lambda c, fs=filters: any(f(c) for f in fs)
        plan.hard_reason = ", ".join(reasons)

    # 3. Named sections ("South Asia", "East Asia and Pacific", "SAR") are boosted.
    for title in sections:
        words = title.split()
        if title.lower() in GENERIC_TITLES or len(words) < 2 or len(title) > 70:
            continue
        if _phrase_in(title, question):
            plan.soft_sections.append(title)
    for title, al in aliases.items():
        for a in al:
            if a.isupper() and re.search(r"(?<![A-Za-z])" + re.escape(a) + r"(?![A-Za-z])", question):
                if title in sections and title not in plan.soft_sections:
                    plan.soft_sections.append(title)

    # 4. Acronyms both ways, for the keyword index only.
    for acr, expansion in glossary.items():
        if _phrase_in(expansion, question):
            plan.bm25_extra += tokenize(acr)
        elif re.search(r"(?<![A-Za-z])" + re.escape(acr) + r"(?![A-Za-z])", question):
            plan.bm25_extra += tokenize(expansion)
    return plan


def _route_document(store: Store, question: str) -> tuple[set[str] | None, str]:
    """Scope to one document when the question uses terms only it contains.

    "iMac", "GPO", "FDsys", "South Asia" each occur in one document and almost
    never in the others; a term found everywhere ("growth") casts no vote.
    """
    votes: dict[str, int] = {}
    evidence: dict[str, list[str]] = {}
    seen = set()
    for term in tokenize(question):
        if term in seen or term.isdigit():
            continue
        seen.add(term)
        hits = store.bm25.df(term)
        if not hits:
            continue
        per_doc: dict[str, int] = {}
        for i in hits:
            d = store.chunks[i]["doc_id"]
            per_doc[d] = per_doc.get(d, 0) + 1
        top = max(per_doc, key=per_doc.get)
        others = sum(v for d, v in per_doc.items() if d != top)
        if per_doc[top] >= 2 and others <= max(1, per_doc[top] // 20):
            votes[top] = votes.get(top, 0) + 1
            evidence.setdefault(top, []).append(term)
    if len(votes) == 1:
        d = next(iter(votes))
        return {d}, f"terms only in this document: {', '.join(evidence[d][:4])}"
    if len(votes) > 1:
        ranked = sorted(votes.items(), key=lambda x: -x[1])
        if ranked[0][1] >= 2 * ranked[1][1]:
            d = ranked[0][0]
            return {d}, f"terms mostly in this document: {', '.join(evidence[d][:4])}"
    return None, ""


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def search(store: Store, llm, question: str, plan: QueryPlan) -> tuple[list[tuple[int, float]], dict]:
    base = store.mask(plan.doc_ids)
    if plan.hard_filter is not None:
        hard = base & store.mask(None, plan.hard_filter)
        if hard.any():
            base = hard
    qvec = llm.embed_query(question)
    tokens = tokenize(question) + plan.bm25_extra
    k = config.CANDIDATES

    lists: list[tuple[str, float, list[tuple[int, float]]]] = []
    dense = store.dense(qvec, base, k)
    lists.append(("dense", 1.0, dense))
    if config.HYBRID:
        lists.append(("bm25", 1.0, store.sparse(tokens, base, k)))
    if plan.soft_sections:
        wanted = set(plan.soft_sections)
        soft = base & store.mask(None, lambda c: any(p in wanted for p in c["section_path"]))
        if soft.any():
            lists.append(("dense@section", 1.5, store.dense(qvec, soft, k // 2)))
            if config.HYBRID:
                lists.append(("bm25@section", 1.5, store.sparse(tokens, soft, k // 2)))
    for n, part in enumerate(plan.parts, start=1):
        # Later parts lean on the first for their subject ("when was *it* deployed").
        if n > 1:
            part = f"{part} ({plan.parts[0].rstrip('?')})"
        pvec = llm.embed_query(part)
        lists.append((f"dense@part{n}", 1.0, store.dense(pvec, base, k // 2)))
        if config.HYBRID:
            lists.append((f"bm25@part{n}", 1.0, store.sparse(tokenize(part) + plan.bm25_extra, base, k // 2)))

    fused: dict[int, float] = {}
    for _, weight, ranked in lists:
        for rank, (idx, _) in enumerate(ranked):
            fused[idx] = fused.get(idx, 0.0) + weight / (RRF_K + rank + 1)
    dense_score = dict(dense)
    order = sorted(fused, key=lambda i: -fused[i])

    # Diversity: near-duplicates (GPO repeats facts on p5 and p21) and page floods.
    kept: list[int] = []
    token_sets: dict[int, set] = {}
    per_page: dict[tuple, int] = {}
    for idx in order:
        c = store.chunks[idx]
        toks = set(tokenize(c["text"]))
        if any(_jaccard(toks, token_sets[j]) >= 0.8 for j in kept):
            continue
        key = (c["doc_id"], c["page"])
        if per_page.get(key, 0) >= 3:
            continue
        per_page[key] = per_page.get(key, 0) + 1
        token_sets[idx] = toks
        kept.append(idx)
        if len(kept) >= max(config.RERANK_TOP, config.TOP_K):
            break

    debug = {
        "lists": {name: [store.chunks[i]["id"] for i, _ in ranked[:10]] for name, _, ranked in lists},
        "best_dense": max(dense_score.values()) if dense_score else 0.0,
    }
    return [(i, dense_score.get(i, 0.0)) for i in kept], debug


def best_window(text: str, question: str, size: int = 900) -> str:
    """The part of a long chunk that best matches the question.

    Showing the reranker only the first N characters hid answers that sit late
    in a chunk (the runbook's rollback command starts ~750 characters in).
    """
    if len(text) <= size:
        return text
    q = set(tokenize(question))
    words = list(re.finditer(r"\S+", text))
    best_start, best_score = 0, -1
    step = max(1, len(words) // 40)
    for i in range(0, len(words), step):
        start = words[i].start()
        window = text[start:start + size]
        score = len(q & set(tokenize(window)))
        if score > best_score:
            best_start, best_score = start, score
    piece = text[best_start:best_start + size]
    return ("… " if best_start else "") + piece + (" …" if best_start + size < len(text) else "")


def rerank(store: Store, llm, question: str, candidates: list[tuple[int, float]]):
    """Returns ([(idx, score 0-3)], ok). ok=False means the reranker could not run."""
    pool = candidates[: config.RERANK_TOP]
    passages = []
    for n, (idx, _) in enumerate(pool):
        c = store.chunks[idx]
        passages.append(f"[{n}] {c['header']}\n{best_window(c['text'], question)}")
    try:
        data = llm.generate_json(
            RERANK_PROMPT.format(question=question, passages="\n\n".join(passages)), RerankResult
        )
        scores = {int(s["id"]): int(s["score"]) for s in data.get("scores", [])}
    except Exception:
        return [(idx, None) for idx, _ in pool], False
    ranked = [(idx, scores.get(n, 0)) for n, (idx, _) in enumerate(pool)]
    ranked.sort(key=lambda x: -(x[1] or 0))  # stable: ties keep fused order
    return ranked, True


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------


def _block(store: Store, idx: int, score) -> dict:
    c = store.chunks[idx]
    return {
        "chunk_id": c["id"], "doc_id": c["doc_id"], "file_name": c["file_name"], "kind": c["kind"],
        "page": c["page"], "page_end": c["page_end"], "page_label": c["page_label"],
        "section_path": c["section_path"], "header": c["header"], "text": c["text"],
        "order": c["order"], "parent_id": c.get("parent_id"), "score": score,
    }


def assemble(store: Store, selected: list[tuple[int, object]], plan: QueryPlan) -> list[dict]:
    blocks = [_block(store, idx, s) for idx, s in selected]

    # Small-to-big: summaries need the whole passage, not one fragment of it.
    parent_hits: dict[str, int] = {}
    for b in blocks:
        if b["parent_id"]:
            parent_hits[b["parent_id"]] = parent_hits.get(b["parent_id"], 0) + 1
    expanded: list[dict] = []
    used_parents = set()
    for b in blocks:
        pid = b["parent_id"]
        if pid and pid in store.parents and (plan.summary or parent_hits[pid] >= 2):
            if pid in used_parents:
                continue
            used_parents.add(pid)
            parent = store.parents[pid]
            b = {**b, "text": parent["text"], "page": parent["page"], "page_end": parent["page_end"],
                 "kind": "text", "expanded": True}
        expanded.append(b)

    # Budget, then reading order so the model sees coherent passages.
    out, used = [], 0
    for b in expanded:
        if used + len(b["text"]) > config.CONTEXT_CHAR_BUDGET and out:
            continue
        out.append(b)
        used += len(b["text"])
    doc_rank = {d: i for i, d in enumerate(dict.fromkeys(b["doc_id"] for b in out))}
    out.sort(key=lambda b: (doc_rank[b["doc_id"]], b["order"]))
    return out


def retrieve(store: Store, llm, question: str, doc_ids: list[str] | None = None) -> Retrieval:
    plan = analyse(store, question, doc_ids)
    plan.parts = split_parts(question)
    candidates, debug = search(store, llm, question, plan)
    debug["plan"] = {
        "doc_ids": sorted(plan.doc_ids) if plan.doc_ids else None, "doc_reason": plan.doc_reason,
        "hard_filter": plan.hard_reason, "soft_sections": plan.soft_sections,
        "bm25_extra": plan.bm25_extra, "parts": plan.parts, "summary": plan.summary, "numeric": plan.numeric,
    }
    if not candidates:
        return Retrieval([], True, plan, debug)

    limit = config.TOP_K if (plan.summary or plan.numeric) else min(config.TOP_K, 6)
    if config.RERANK == "llm":
        ranked, ok = rerank(store, llm, question, candidates)
        debug["rerank"] = [(store.chunks[i]["id"], s) for i, s in ranked]
        if ok:
            good = [(i, s) for i, s in ranked if s is not None and s >= 2]
            if not good:
                return Retrieval([], True, plan, debug)
            return Retrieval(assemble(store, good[:limit], plan), False, plan, debug)
        # Reranker unavailable: fall through to the fused order and cosine gate.
    if debug["best_dense"] < config.RELEVANCE_THRESHOLD:
        return Retrieval([], True, plan, debug)
    selected = [(i, round(s, 3)) for i, s in candidates[:limit]]
    return Retrieval(assemble(store, selected, plan), False, plan, debug)
