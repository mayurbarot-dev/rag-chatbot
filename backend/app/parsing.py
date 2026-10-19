"""PDF parsing with layout: lines carrying font metadata, repaired text, page labels.

PyMuPDF is used instead of pypdf because the later stages need span-level
font size, weight and superscript flags, bounding boxes, drawing/image counts
and page rendering. pypdf also garbled apostrophes in the GPO report.
"""

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool
    font: str
    block: int
    refs: list[int] = field(default_factory=list)
    role: str = "body"  # body | heading | caption | table | footnote | box
    page: int = 0

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass
class Page:
    number: int  # 1-based PDF page index
    width: float
    height: float
    lines: list[Line]
    label: str
    chapter: str | None = None
    skip: str | None = None  # toc | abbreviations | blank
    drawings: int = 0
    image_ratio: float = 0.0
    tables: list = field(default_factory=list)  # tables.Table, filled later
    printed: str | None = None  # printed page number harvested from the margins

    @property
    def text_chars(self) -> int:
        return sum(len(line.text) for line in self.lines)


@dataclass
class ParsedDoc:
    file_name: str
    doc_id: str
    sha256: str
    title: str
    pages: list[Page]
    body_size: float
    body_font: str
    toc: list[tuple[int, str, int]]
    glossary: dict[str, str]
    th_repair: bool

    @property
    def page_count(self) -> int:
        return len(self.pages)


# --------------------------------------------------------------------------
# Text repair
# --------------------------------------------------------------------------

# Some fonts map the "Th" ligature glyph to a bare "T" (GEP: "Te global economy",
# 186 times). Only capitalised words are affected; lowercase "th" is intact.
TH_WORDS = {
    "Te": "The", "Tis": "This", "Tat": "That", "Tese": "These", "Tose": "Those",
    "Tey": "They", "Teir": "Their", "Tere": "There", "Tus": "Thus",
    "Tough": "Though", "Trough": "Through", "Tird": "Third", "Terefore": "Therefore",
    "Tousand": "Thousand", "Tailand": "Thailand", "Teory": "Theory",
    "Temselves": "Themselves", "Tereafter": "Thereafter", "Tereby": "Thereby",
    "Tem": "Them", "Tursday": "Thursday", "Tirty": "Thirty",
}
TH_RE = re.compile(r"\b(" + "|".join(sorted(TH_WORDS, key=len, reverse=True)) + r")\b")
TH_SIGNAL_RE = re.compile(r"\b(?:Te|Tis|Tese|Tey)\b")


def repair_text(text: str, th_repair: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "").replace("\x00", "").replace("�", "'")
    if th_repair:
        text = TH_RE.sub(lambda m: TH_WORDS[m.group(1)], text)
    return re.sub(r"[ \t\r\f\v ]+", " ", text).strip()


def join_lines(lines: list[str]) -> str:
    """Join wrapped lines into one paragraph, removing end-of-line hyphenation.

    "confi-" + "dence" -> "confidence", but "low-" + "Carbon" and "2027-" + "28"
    keep their hyphen.
    """
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif re.search(r"[a-z]-$", out) and re.match(r"[a-z]", line):
            out = out[:-1] + line
        else:
            out += " " + line
    return out


# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

ROMAN_RE = re.compile(r"^[ivxlcdm]{1,7}$", re.IGNORECASE)
PAGE_NUM_RE = re.compile(r"^\d{1,4}$")
DOT_LEADER_RE = re.compile(r"(\.\s?){4,}\s*[\dixvlc]*\s*$|\.{3,}\s*\d+\s*$")
TOC_TITLES = {
    "contents", "table of contents", "summary of contents", "list of figures",
    "list of tables", "list of boxes",
}
ABBREV_TITLES = {"abbreviations", "acronyms", "abbreviations and acronyms", "list of abbreviations"}
ACRONYM_RE = re.compile(r"^(?=(?:.*[A-Z]){2})[A-Z][A-Za-z0-9&/.-]{1,11}$")
CHAPTER_RE = re.compile(r"\bchapter\s+(\d+)\b", re.IGNORECASE)


def _key(text: str) -> str:
    """Normalisation used to spot running headers: letters only, lowercase."""
    return re.sub(r"[^a-z]", "", text.lower())


def _decode_label(label: str) -> str:
    # PyMuPDF sometimes returns UTF-16 escapes such as "<FEFF0063>1" for "c1".
    return re.sub(r"<FEFF([0-9A-Fa-f]{4})>", lambda m: chr(int(m.group(1), 16)), label or "")


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def _is_bold(span) -> bool:
    font = span.get("font", "")
    return bool(span["flags"] & 16) or any(w in font for w in ("Bold", "Semibold", "Heavy", "Black"))


def _raw_lines(page) -> list[Line]:
    lines: list[Line] = []
    data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)
    for b_no, block in enumerate(data["blocks"]):
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            main = max(spans, key=lambda s: len(s["text"].strip()))
            main_size = main["size"]
            text_parts: list[str] = []
            refs: list[int] = []
            prev_x1 = None
            for i, span in enumerate(spans):
                raw = span["text"]
                token = raw.strip()
                is_sup = bool(span["flags"] & 1) or (
                    span["size"] <= 0.75 * main_size and span["bbox"][3] < line["bbox"][3] - 0.2 * main_size
                )
                if i > 0 and is_sup and re.fullmatch(r"\d{1,3}(,\d{1,3})*", token):
                    refs.extend(int(n) for n in token.split(","))
                    # The marker often stood in for the word gap ("suppliers.12 By").
                    text_parts.append(" ")
                    prev_x1 = span["bbox"][2]
                    continue
                # Some PDFs position words without a space glyph between spans.
                if (prev_x1 is not None and text_parts and not text_parts[-1].endswith(" ")
                        and not raw.startswith(" ") and span["bbox"][0] - prev_x1 > 0.18 * span["size"]):
                    text_parts.append(" ")
                text_parts.append(raw)
                prev_x1 = span["bbox"][2]
            text = "".join(text_parts)
            if not text.strip():
                continue
            body = [s for s in spans if not (s["flags"] & 1)] or spans
            chars = sum(len(s["text"]) for s in body)
            size = sum(s["size"] * len(s["text"]) for s in body) / max(chars, 1)
            bold = all(_is_bold(s) for s in body if s["text"].strip())
            x0, y0, x1, y1 = line["bbox"]
            lines.append(
                Line(text=text, x0=x0, y0=y0, x1=x1, y1=y1, size=round(size, 2),
                     bold=bold, font=main.get("font", ""), block=b_no, refs=refs)
            )
    return lines


def _merge_drop_caps(lines: list[Line]) -> list[Line]:
    """'T' (60pt) + 'he U.S. Government...' -> 'The U.S. Government...'."""
    out: list[Line] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if (nxt and len(line.text.strip()) == 1 and line.text.strip().isupper()
                and line.size >= 2.5 * nxt.size and nxt.text[:1].islower()):
            nxt.text = line.text.strip() + nxt.text
            nxt.x0 = min(nxt.x0, line.x0)
            i += 1
            continue
        out.append(line)
        i += 1
    return out


def _visual_signals(page) -> tuple[int, float]:
    try:
        drawings = len(page.get_cdrawings())
    except Exception:
        drawings = 0
    area = page.rect.width * page.rect.height or 1.0
    covered = 0.0
    try:
        for info in page.get_image_info():
            x0, y0, x1, y1 = info["bbox"]
            covered += max(0.0, x1 - x0) * max(0.0, y1 - y0)
    except Exception:
        pass
    return drawings, min(1.0, covered / area)


def _doc_title(doc, pages: list[Page], file_name: str) -> str:
    meta = (doc.metadata or {}).get("title", "") or ""
    meta = meta.strip()
    if 4 <= len(meta) <= 120 and not meta.lower().endswith((".pdf", ".doc", ".docx", ".indd")) \
            and not re.match(r"^(untitled|microsoft|document\d*)", meta, re.IGNORECASE):
        return meta
    # Otherwise the largest text on the first two pages -- but only when it is
    # visibly a title (bigger than the body) and short; else the file name.
    sizes = Counter()
    for page in pages[:3]:
        for line in page.lines:
            sizes[round(line.size)] += len(line.text)
    body = sizes.most_common(1)[0][0] if sizes else 0
    best: list[Line] = []
    best_size = 0.0
    for page in pages[:2]:
        for line in page.lines:
            if len(re.sub(r"[^A-Za-z]", "", line.text)) < 3:
                continue
            if line.size > best_size + 0.5:
                best, best_size = [line], line.size
            elif abs(line.size - best_size) <= 0.5 and best and line.page == best[0].page and len(best) < 3:
                best.append(line)
    if best and best_size >= body + 2:
        return " ".join(line.text for line in best)[:120]
    return re.sub(r"[_-]+", " ", file_name.rsplit(".", 1)[0])


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def parse_pdf(data: bytes, file_name: str) -> ParsedDoc:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"'{file_name}' could not be opened as a PDF: {exc}") from exc
    if doc.needs_pass:
        raise ValueError(f"'{file_name}' is password-protected.")

    sha = hashlib.sha256(data).hexdigest()
    raw_pages: list[tuple] = []
    signal = 0
    for page in doc:
        lines = _raw_lines(page)
        signal += sum(len(TH_SIGNAL_RE.findall(line.text)) for line in lines)
        drawings, image_ratio = _visual_signals(page)
        raw_pages.append((page, lines, drawings, image_ratio))

    th_repair = signal >= 5
    pages: list[Page] = []
    for page, lines, drawings, image_ratio in raw_pages:
        for line in lines:
            line.text = repair_text(line.text, th_repair)
            line.page = page.number + 1
        lines = _merge_drop_caps([line for line in lines if line.text])
        label = _decode_label(page.get_label()) if page.get_label() else ""
        pages.append(
            Page(number=page.number + 1, width=page.rect.width, height=page.rect.height,
                 lines=lines, label=label, drawings=drawings, image_ratio=image_ratio)
        )

    _strip_running_lines(pages)

    body_size, body_font = _body_metrics(pages)
    glossary: dict[str, str] = {}
    for page in pages:
        _detect_skip(page, glossary)

    toc = _clean_toc(doc.get_toc(simple=True), len(pages))
    title = _doc_title(doc, pages, file_name)
    stem = _slug(file_name.rsplit(".", 1)[0])[:16] or "doc"

    total = sum(p.text_chars for p in pages)
    if total < 50:
        raise ValueError(
            f"No readable text found in '{file_name}'. It may be a scanned or "
            "image-only PDF, which would need OCR before it can be indexed."
        )

    return ParsedDoc(
        file_name=file_name, doc_id=f"{stem}-{sha[:6]}", sha256=sha, title=title,
        pages=pages, body_size=body_size, body_font=body_font, toc=toc,
        glossary=glossary, th_repair=th_repair,
    )


# --------------------------------------------------------------------------
# Running headers / footers and page labels
# --------------------------------------------------------------------------


def _strip_running_lines(pages: list[Page]) -> None:
    """Remove repeated margin lines, harvesting printed page numbers and chapters first."""
    n = len(pages)
    counts: Counter[str] = Counter()
    margin: list[list[Line]] = []
    for page in pages:
        top, bottom = page.height * 0.10, page.height * 0.88
        lines = [line for line in page.lines if line.y1 <= top or line.y0 >= bottom]
        margin.append(lines)
        for key in {_key(line.text) for line in lines if len(_key(line.text)) >= 4}:
            counts[key] += 1
    # Odd and even pages often carry different banners, so each variant may
    # only appear on half of the pages.
    threshold = max(3, int(0.1 * n))

    for page, lines in zip(pages, margin):
        drop = set()
        printed = None
        tail_number = None
        for line in lines:
            text = line.text.strip()
            key = _key(text)
            if PAGE_NUM_RE.match(text) or (ROMAN_RE.match(text) and len(text) <= 6):
                printed = printed or text
                drop.add(id(line))
            elif len(key) >= 4 and counts[key] >= threshold:
                m = CHAPTER_RE.search(text)
                if m:
                    page.chapter = m.group(1)
                else:
                    tail = re.search(r"\s(\d{1,4})$", text)
                    if tail:
                        tail_number = tail_number or tail.group(1)
                drop.add(id(line))
        page.lines = [line for line in page.lines if id(line) not in drop]
        page.printed = printed or tail_number

    _assign_labels(pages)


def _assign_labels(pages: list[Page]) -> None:
    """Accept a harvested printed number only when it fits the document's numbering.

    Arabic page numbers keep a constant offset from the PDF index within a run, so
    a stray margin digit (a footnote marker, a call-out) shows up as an outlier.
    Roman numerals are accepted as they are (front matter).
    """
    offsets = {}
    for page in pages:
        value = getattr(page, "printed", None)
        if value and value.isdigit():
            offsets[page.number] = int(value) - page.number
    common = Counter(offsets.values()).most_common(1)
    mode = common[0][0] if common else None
    for page in pages:
        value = getattr(page, "printed", None)
        ok = False
        if value and not value.isdigit():
            ok = True
        elif value:
            off = offsets[page.number]
            ok = off == mode or offsets.get(page.number - 1) == off or offsets.get(page.number + 1) == off
        if ok:
            page.label = value
        elif not page.label or page.label == str(page.number):
            page.label = str(page.number)


def _body_metrics(pages: list[Page]) -> tuple[float, str]:
    sizes: Counter[float] = Counter()
    fonts: Counter[str] = Counter()
    for page in pages:
        for line in page.lines:
            sizes[round(line.size * 2) / 2] += len(line.text)
            fonts[line.font] += len(line.text)
    body_size = sizes.most_common(1)[0][0] if sizes else 10.0
    body_font = fonts.most_common(1)[0][0] if fonts else ""
    return body_size, body_font


def font_family(font: str) -> str:
    return re.split(r"[-,]", font or "", maxsplit=1)[0].lower()


# --------------------------------------------------------------------------
# Front matter
# --------------------------------------------------------------------------


def _detect_skip(page: Page, glossary: dict[str, str]) -> None:
    lines = page.lines
    if page.text_chars < 30:
        page.skip = "blank"
        return

    first = next((line.text.strip().lower() for line in lines if len(line.text.strip()) > 3), "")
    leaders = sum(1 for line in lines if DOT_LEADER_RE.search(line.text))
    if first in TOC_TITLES or (leaders >= 4 and leaders >= 0.25 * len(lines)):
        page.skip = "toc"
        return

    titles = [line for line in lines if line.text.strip().lower() in ABBREV_TITLES]
    acronyms = [line for line in lines if ACRONYM_RE.match(line.text.strip())]
    if titles and len(acronyms) >= 8:
        glossary.update(_parse_glossary(lines, acronyms))
        page.skip = "abbreviations"


def _parse_glossary(lines: list[Line], acronyms: list[Line]) -> dict[str, str]:
    """Pair each acronym with the expansion printed on the same baseline to its right."""
    out: dict[str, str] = {}
    right_edge = max(a.x1 for a in acronyms)
    candidates = [
        line for line in lines
        if line.x0 >= right_edge - 2 and line not in acronyms and len(line.text) > 2
    ]
    for acr in acronyms:
        tol = 0.6 * acr.size
        match = min(
            (c for c in candidates if abs(c.yc - acr.yc) <= tol),
            key=lambda c: abs(c.yc - acr.yc),
            default=None,
        )
        if match:
            out[acr.text.strip()] = match.text.strip()
    return out


# --------------------------------------------------------------------------
# Outline
# --------------------------------------------------------------------------


def clean_title(title: str) -> str:
    title = repair_text(title.replace("\n", " ").replace("\r", " "))
    title = re.sub(r"^[-–—\s]+", "", title)
    title = re.sub(r"\s+--\s+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def _clean_toc(raw: list, page_count: int) -> list[tuple[int, str, int]]:
    """Drop outline noise: file names, sentence-length entries and repeated banners."""
    entries = []
    for level, title, page in raw:
        title = clean_title(title)
        if not title or page is None or page < 1 or page > page_count:
            continue
        if title.lower().endswith(".pdf") or len(title) > 100 or title.endswith((":", ",")):
            continue
        # Subtitle fragments the outline split off ("and Changes in...", "As of
        # September 30, 2018") are not sections of their own.
        if title[0].islower() or re.match(r"^(as of|for the (fiscal )?years? ended)\b", title, re.I):
            continue
        entries.append((level, title, page))
    repeated = {t for t, c in Counter(t for _, t, _ in entries).items() if c >= 3}
    entries = [e for e in entries if e[1] not in repeated]
    if not entries:
        return []
    shift = min(level for level, _, _ in entries) - 1
    return [(level - shift, title, page) for level, title, page in entries]
