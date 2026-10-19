"""Chart and infographic transcription with a multimodal model.

Text extraction cannot recover what a chart encodes in position and colour:
the iMac's p2 waterfall comes out as a jumble of labels ("–99 kg", "346 kg",
"Electricity for manufacturing" three times). Pages that look visual are
rendered and transcribed into structured text, which is indexed as its own
`visual` chunk alongside the page's ordinary text.

Results are cached by file hash, page, model and prompt version, so
re-indexing, re-uploading and repeated eval runs never pay twice.
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pymupdf
from pydantic import BaseModel

from . import config
from .parsing import ParsedDoc
from .structure import CAPTION_RE

log = logging.getLogger("uvicorn.error")

PROMPT_VERSION = "v1"

VISION_PROMPT = """You are transcribing one page of a PDF report for a search index.

Extract ONLY the information carried by charts, graphs, infographics, icon \
call-outs, diagrams and data tables on this page. For each such element give:
- type: chart | table | infographic | diagram | callout
- title: the element's title exactly as printed ("" if none)
- summary: one or two sentences saying what it shows, using only what is printed
- data: every data label and value exactly as printed, with units, each paired \
with the category, series, year or configuration it belongs to

Rules:
- Do not compute, estimate or infer any number that is not printed.
- Copy footnote markers as printed (e.g. "Baseline emissions 7").
- If a value cannot be read, use "unreadable".
- Ignore photographs, logos, decorative graphics and ordinary paragraph text.
- If the page has no such element, return an empty list.

Text extracted from this page, for spelling reference only (it may be jumbled):
{hint}"""


class DataPoint(BaseModel):
    label: str
    value: str


class Visual(BaseModel):
    type: str
    title: str
    summary: str
    data: list[DataPoint]


class PageVisuals(BaseModel):
    visuals: list[Visual]


def score_page(page, body_size: float) -> float:
    """Heuristic: how likely is it that this page carries data only an image shows?"""
    if page.skip:
        return 0.0
    if page.number <= 2 and page.text_chars < 300:
        return 0.0  # a cover: artwork, not data
    lines = [l for l in page.lines if l.role in ("body", "caption", "heading")]
    score = 0.0
    if page.drawings >= 40:
        score += 2.0
    if page.drawings >= 150:
        score += 1.0
    if page.image_ratio >= 0.2:
        score += 1.5
    if any(CAPTION_RE.match(l.text.strip()) and l.text.strip().upper().startswith(("FIGURE", "CHART", "MAP"))
           for l in page.lines):
        score += 2.0
    if lines:
        short = sum(1 for l in lines if len(l.text.strip()) <= 25)
        if short / len(lines) >= 0.4:
            score += 1.5
        numeric_labels = sum(1 for l in lines if len(l.text.strip()) <= 14 and re.search(r"\d", l.text))
        if numeric_labels >= 3:
            score += 1.0
    if page.tables and score < 4:
        score -= 2.0  # its numbers were already rebuilt as a table
    return score


def select_pages(doc: ParsedDoc) -> list[int]:
    if config.VISION_MODE == "off":
        return []
    scored = [(score_page(p, doc.body_size), p.number) for p in doc.pages if not p.skip]
    if config.VISION_MODE == "all":
        return [n for _, n in scored]
    picked = sorted((s, n) for s, n in scored if s >= 3.0)
    picked = sorted(picked, key=lambda x: (-x[0], x[1]))[: config.VISION_MAX_PAGES]
    return sorted(n for _, n in picked)


def _cache_path(sha: str, page: int):
    model = re.sub(r"[^A-Za-z0-9.-]+", "_", config.VISION_MODEL)
    return config.CACHE_DIR / "vision" / f"{sha[:16]}_{page}_{model}_{PROMPT_VERSION}_{config.VISION_DPI}.json"


def serialize(visual: dict) -> str:
    parts = []
    kind = (visual.get("type") or "visual").strip()
    title = (visual.get("title") or "").strip()
    summary = (visual.get("summary") or "").strip()
    parts.append(f"[{kind}{': ' + title if title else ''}]")
    if summary:
        parts.append(summary)
    data = [d for d in visual.get("data") or [] if (d.get("value") or "").strip()]
    if data:
        parts.append("Values: " + "; ".join(f"{d['label'].strip()} = {d['value'].strip()}" for d in data))
    return " ".join(parts)


def transcribe(llm, pdf_bytes: bytes, doc: ParsedDoc, pages: list[int], progress=None) -> tuple[dict, list[str]]:
    """Returns ({page: [text, ...]}, warnings)."""
    results: dict[int, list[str]] = {}
    warnings: list[str] = []
    if not pages:
        return results, warnings
    pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    jobs = {}
    for number in pages:
        cached = _cache_path(doc.sha256, number)
        if cached.exists():
            try:
                data = json.loads(cached.read_text(encoding="utf-8"))
                results[number] = [serialize(v) for v in data.get("visuals", [])]
                continue
            except (OSError, json.JSONDecodeError):
                pass
        page = pdf[number - 1]
        png = page.get_pixmap(dpi=config.VISION_DPI).tobytes("png")
        hint = " | ".join(l.text for l in doc.pages[number - 1].lines)[:2500]
        jobs[number] = (png, hint)

    done = len(results)
    if progress:
        progress(done, len(pages))

    def run(number: int):
        png, hint = jobs[number]
        return number, llm.describe_image(png, VISION_PROMPT.format(hint=hint), PageVisuals)

    with ThreadPoolExecutor(max_workers=max(1, config.VISION_CONCURRENCY)) as pool:
        futures = [pool.submit(run, n) for n in jobs]
        for fut in as_completed(futures):
            try:
                number, data = fut.result()
            except Exception as exc:  # one bad page must not fail the upload
                warnings.append(f"Vision transcription failed on a page: {exc}")
                log.warning("vision failed: %s", exc)
            else:
                path = _cache_path(doc.sha256, number)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                results[number] = [serialize(v) for v in data.get("visuals", [])]
            done += 1
            if progress:
                progress(done, len(pages))
    return results, warnings
