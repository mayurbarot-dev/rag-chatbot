"""Section tree, heading detection, box overlays and the document profile.

Every content line ends up with a section id. The tree is seeded from the PDF
outline when one exists, but outline destinations are only trusted as hints:
GEP's outline sends "South Asia" to p95, which is the previous region's
forecast table; the heading itself is at the top of p96. Each entry is snapped
to the matching heading found near its page.
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .parsing import Line, ParsedDoc, font_family

CAPTION_RE = re.compile(r"^(?:FIGURE|Figure|TABLE|Table|CHART|Chart|MAP|Map)\s+[A-Z]?\d+(?:\.\d+)*\b")
BOX_RE = re.compile(r"^(BOX|Box)\s+([A-Z]?\d+(?:\.\d+)*)\s+(.+?)(?:\s*\(continued\))?\s*$")
TOP_LEVEL_RE = re.compile(
    r"^(ch\.?|chapter|part|annex|appendix|statistical annex|executive summary|foreword|"
    r"preface|references|glossary|overview|summary|introduction)\b",
    re.IGNORECASE,
)
CHAPTER_TITLE_RE = re.compile(r"^(?:ch\.?|chapter)\s*(\d+)\.?\s*(.*)$", re.IGNORECASE)
SKIP_TOC = {"contents", "summary of contents", "table of contents"}
GENERIC_TITLES = {
    "recent developments", "outlook", "risks", "summary", "introduction", "conclusion",
    "conclusions", "references", "main findings", "policy challenges", "overview", "notes",
    "endnotes", "annex", "background", "methodology", "funding", "definitions",
}


@dataclass
class Section:
    id: int
    title: str
    level: int
    kind: str  # outline | heading | box | root
    path: list[str]
    page: int
    aliases: list[str] = field(default_factory=list)


@dataclass
class Structure:
    sections: list[Section]
    profile: str
    signals: dict


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def display_title(title: str) -> str:
    m = CHAPTER_TITLE_RE.match(title)
    if m:
        rest = m.group(2).strip()
        return f"Chapter {m.group(1)}" + (f". {rest}" if rest else "")
    return title


# --------------------------------------------------------------------------
# Headings
# --------------------------------------------------------------------------


def _heading_like_text(text: str, whole: bool = True) -> bool:
    """`whole=False` checks one line of a possibly wrapped heading."""
    t = text.strip()
    if not (2 <= len(t) <= 100):
        return False
    letters = sum(c.isalpha() for c in t)
    if letters < 2 or letters / len(t) < 0.5:
        return False
    if t.startswith(("•", "-", "–", "(")) or CAPTION_RE.match(t) or BOX_RE.match(t):
        return False
    if whole and (len(t) < 3 or letters < 3 or t.endswith((".", ",", ";", ":"))):
        return False
    return True


def _page_body(page, doc: ParsedDoc) -> tuple[float, str]:
    """The page's own body style. A report can mix styles: GPO's narrative is
    9.75pt Helvetica while its (longer) financial section is 12pt Times, so the
    document-wide mode would hide every narrative heading."""
    sizes: dict[float, int] = {}
    fonts: dict[str, int] = {}
    for line in page.lines:
        if line.role != "body":
            continue
        key = round(line.size * 2) / 2
        sizes[key] = sizes.get(key, 0) + len(line.text)
        fonts[line.font] = fonts.get(line.font, 0) + len(line.text)
    if sum(sizes.values()) < 400:
        return doc.body_size, font_family(doc.body_font)
    return max(sizes, key=sizes.get), font_family(max(fonts, key=fonts.get))


def detect_headings(doc: ParsedDoc) -> None:
    """Mark heading lines (role='heading') and give each a rank (0 = biggest)."""
    for page in doc.pages:
        if page.skip:
            continue
        body, family = _page_body(page, doc)
        blocks: dict[int, list[Line]] = {}
        for line in page.lines:
            blocks.setdefault(line.block, []).append(line)
        # Captions and their wrapped continuation lines are never headings.
        for line in page.lines:
            if line.role == "body" and CAPTION_RE.match(line.text.strip()):
                line.role = "caption"
                for other in blocks[line.block]:
                    if other is not line and other.role == "body" and abs(other.size - line.size) < 0.6 \
                            and other.bold == line.bold and other.y0 >= line.y0:
                        other.role = "caption"
        candidates: set[int] = set()
        for line in page.lines:
            if line.role != "body" or not _heading_like_text(line.text, whole=False):
                continue
            block = blocks[line.block]
            body_lines = [l for l in block if l.role == "body"]
            big = line.size >= body + 1.5
            # A heading stands alone: a bold lead-in inside a paragraph block is not one.
            if len(body_lines) > (4 if big else 3):
                continue
            bold = line.bold and line.size >= body - 0.6
            other_font = (
                font_family(line.font) != family and line.size >= body - 0.3
                and len(body_lines) <= 2 and len(line.text) <= 60
            )
            if big or bold or other_font:
                candidates.add(id(line))
                line.heading_weight = round(line.size + (0.5 if line.bold else 0) + (0.25 if big else 0), 1)

        # Wrapped headings: consecutive candidate lines with the same style and
        # left edge become one ("Taking responsibility for" + "our products at
        # every stage"), whether or not the PDF put them in one text block.
        # The joined text is then validated as a whole, so a wrapped subtitle
        # sentence ending in a full stop is rejected.
        merged: list[Line] = []
        groups: list[list[Line]] = []
        for line in page.lines:
            prev = merged[-1] if merged else None
            if (prev is not None and id(line) in candidates and groups and groups[-1][-1] is prev
                    and id(prev) in candidates and abs(prev.size - line.size) < 0.6
                    and prev.bold == line.bold and abs(prev.x0 - line.x0) <= 6
                    and -0.5 * line.size <= line.y0 - prev.y1 < 0.8 * line.size):
                groups[-1].append(line)
                prev.text = f"{prev.text} {line.text}".strip()
                prev.y1 = line.y1
                prev.refs.extend(r for r in line.refs if r not in prev.refs)
                continue
            merged.append(line)
            groups.append([line])
        page.lines = merged
        for line in page.lines:
            if id(line) in candidates:
                if _heading_like_text(line.text) and len(line.text) <= 100:
                    line.role = "heading"
                else:
                    line.role = "body"
        for line in page.lines:
            if line.role == "heading" and not hasattr(line, "heading_weight"):
                line.heading_weight = round(line.size + (0.5 if line.bold else 0) + 0.25, 1)

    weights = sorted({getattr(l, "heading_weight", 0) for p in doc.pages for l in p.lines if l.role == "heading"},
                     reverse=True)
    rank = {w: min(i, 3) for i, w in enumerate(weights)}
    for page in doc.pages:
        for line in page.lines:
            if line.role == "heading":
                line.heading_rank = rank.get(getattr(line, "heading_weight", 0), 3)


# --------------------------------------------------------------------------
# Outline handling
# --------------------------------------------------------------------------


def _prepare_outline(doc: ParsedDoc) -> list[tuple[int, str, int]]:
    entries = [(lvl, t, p) for lvl, t, p in doc.toc if t.lower() not in SKIP_TOC and not BOX_RE.match(t)]
    if not entries:
        return []
    levels = {lvl for lvl, _, _ in entries}
    if len(levels) == 1:
        # A flat outline ("Ch. 2", "East Asia and Pacific", "South Asia", ...):
        # nest non-chapter entries under the preceding chapter-like entry.
        out = []
        seen_top = False
        for _, title, page in entries:
            if TOP_LEVEL_RE.match(title):
                out.append((1, title, page))
                seen_top = True
            else:
                out.append((2 if seen_top else 1, title, page))
        entries = out
    return entries


def _find_anchor(doc: ParsedDoc, title: str, page_no: int) -> tuple[int, int] | None:
    """Locate the heading line an outline entry points at, near its page."""
    target = _norm(display_title(title))
    alt = _norm(title)
    best = None
    best_score = 0.0
    for p in (page_no, page_no + 1, page_no - 1, page_no + 2):
        if p < 1 or p > len(doc.pages):
            continue
        page = doc.pages[p - 1]
        for idx, line in enumerate(page.lines):
            text = _norm(line.text)
            if not text or len(text) < 3:
                continue
            nxt = _norm(page.lines[idx + 1].text) if idx + 1 < len(page.lines) else ""
            for cand in (text, f"{text} {nxt}".strip()):
                score = max(SequenceMatcher(None, cand, target).ratio(),
                            SequenceMatcher(None, cand, alt).ratio())
                if len(cand) >= 10 and (target.startswith(cand) or alt.startswith(cand)):
                    score = max(score, 0.85)
                if line.role == "heading" or line.size >= doc.body_size + 1.5:
                    score += 0.1
                if p != page_no:
                    score -= 0.03
                if score > best_score:
                    best, best_score = (p, idx), score
    return best if best_score >= 0.82 else None


# --------------------------------------------------------------------------
# Walk
# --------------------------------------------------------------------------


def build_structure(doc: ParsedDoc) -> Structure:
    detect_headings(doc)
    sections: list[Section] = [Section(0, doc.title, 0, "root", [], 1)]

    outline = _prepare_outline(doc)
    anchors: dict[tuple[int, int], tuple[int, str]] = {}
    page_events: dict[int, list[tuple[int, str]]] = {}
    for level, title, page in outline:
        hit = _find_anchor(doc, title, page)
        if hit and hit not in anchors:
            anchors[hit] = (level, title)
        else:
            page_events.setdefault(page, []).append((level, title))

    stack: list[Section] = []
    box_saved: list[Section] | None = None
    box: Section | None = None

    def push(title: str, level: int, kind: str, page: int) -> Section:
        sec = Section(len(sections), display_title(title), level, kind,
                      [s.title for s in stack] + [display_title(title)], page)
        sections.append(sec)
        stack.append(sec)
        return sec

    def outline_event(level: int, title: str, page: int):
        nonlocal box, box_saved
        if box is not None:
            stack[:] = box_saved or []
            box, box_saved = None, None
        while stack and (stack[-1].kind == "heading" or stack[-1].level >= level):
            stack.pop()
        push(title, level, "outline", page)

    def current() -> int:
        return stack[-1].id if stack else 0

    for page in doc.pages:
        for level, title in page_events.get(page.number, []):
            outline_event(level, title, page.number)
        has_box_marker = any(
            BOX_RE.match(l.text.strip()) and l.y0 < page.height * 0.25 for l in page.lines
        )
        if box is not None and not has_box_marker and not page.skip:
            stack[:] = box_saved or []
            box, box_saved = None, None

        for idx, line in enumerate(page.lines):
            key = (page.number, idx)
            if key in anchors:
                level, title = anchors[key]
                outline_event(level, title, page.number)
                line.role = "heading"
                line.section = current()
                continue
            m = BOX_RE.match(line.text.strip())
            if m and line.y0 < page.height * 0.25 and line.role in ("body", "heading"):
                title = f"Box {m.group(2)} {m.group(3).strip()}"
                if box is None or box.title != title:
                    if box is None:
                        box_saved = list(stack)
                    else:
                        stack[:] = box_saved or []
                    base = [s for s in stack if s.kind == "outline" and s.level <= 1] or stack[:1]
                    stack[:] = base
                    box = push(title, (base[-1].level + 1) if base else 1, "box", page.number)
                line.role = "box"
                line.section = current()
                continue
            if line.role == "heading" and not page.skip:
                rank = getattr(line, "heading_rank", 3)
                repeats_title = any(
                    _norm(line.text) and _norm(line.text) in _norm(s.title) and page.number - s.page <= 2
                    for s in stack if s.kind in ("outline", "box")
                )
                if repeats_title:
                    # A divider page repeating the chapter title ("A RISING CHALLENGE").
                    line.role = "body"
                    line.section = current()
                    continue
                if box is not None and line.size >= doc.body_size + 3:
                    stack[:] = box_saved or []
                    box, box_saved = None, None
                while stack and stack[-1].kind == "heading" and getattr(stack[-1], "rank", 3) >= rank:
                    stack.pop()
                sec = push(line.text.strip(), (stack[-1].level + 1) if stack else 1, "heading", page.number)
                sec.rank = rank
            line.section = current()

    _add_aliases(sections, doc.glossary)
    profile, signals = _profile(doc, outline)
    return Structure(sections=sections, profile=profile, signals=signals)


def _add_aliases(sections: list[Section], glossary: dict[str, str]) -> None:
    by_expansion = {_norm(v): k for k, v in glossary.items()}
    for sec in sections:
        aliases = []
        acr = by_expansion.get(_norm(sec.title))
        if acr:
            aliases.append(acr)
        m = re.match(r"^Chapter (\d+)", sec.title)
        if m:
            aliases += [f"Chapter {m.group(1)}", f"Ch. {m.group(1)}"]
        m = re.match(r"^Box ([A-Z]?\d+(?:\.\d+)*)", sec.title)
        if m:
            aliases.append(f"Box {m.group(1)}")
        sec.aliases = aliases


def _profile(doc: ParsedDoc, outline) -> tuple[str, dict]:
    pages = [p for p in doc.pages if not p.skip]
    visual = sum(1 for p in pages if p.drawings >= 40 or p.image_ratio >= 0.25)
    chapters = sum(1 for _, t, _ in outline if CHAPTER_TITLE_RE.match(t))
    tables = sum(len(p.tables) for p in doc.pages)
    signals = {
        "pages": doc.page_count, "outline_entries": len(doc.toc), "chapters": chapters,
        "visual_pages": visual, "tables": tables, "glossary_terms": len(doc.glossary),
    }
    if doc.page_count <= 40 and visual >= 0.3 * max(len(pages), 1):
        return "visual-dense", signals
    if doc.page_count >= 100 or chapters >= 3:
        return "long-structured", signals
    return "narrative", signals
