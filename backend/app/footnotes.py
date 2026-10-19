"""Footnote definitions and the markers that cite them.

Markers come from parsing: superscript numeric spans are removed from the line
text and recorded on Line.refs ("ENERGY STAR.18" -> "ENERGY STAR.", refs=[18]).
This module finds the definitions -- an Endnotes section (iMac pp. 11-12) or
small numbered notes at the foot of a page -- so chunking can attach the note
to every chunk that cites it.
"""

import re
from dataclasses import dataclass, field

from .parsing import Line, ParsedDoc

ENDNOTE_TITLES = {"endnotes", "end notes", "notes", "footnotes"}
NOTE_START = re.compile(r"^(\d{1,3})(?:\.|\s+|(?=[A-Z(“\"]))\s*(\S.*)$")


@dataclass
class Footnote:
    number: int
    text: str
    page: int
    cited_on: list[int] = field(default_factory=list)


def _endnote_pages(doc: ParsedDoc) -> set[int]:
    pages = set()
    for page in doc.pages:
        top = page.height * 0.2
        for line in page.lines:
            if (line.y0 <= top and line.text.strip().lower() in ENDNOTE_TITLES
                    and line.size >= doc.body_size + 1.5):
                pages.add(page.number)
                line.role = "heading"
                break
    return pages


def extract_footnotes(doc: ParsedDoc) -> dict[str, Footnote]:
    """Keys are "e:<n>" for endnotes and "p<page>:<n>" for page-bottom notes."""
    notes: dict[str, Footnote] = {}
    endnote_pages = _endnote_pages(doc)

    # Endnote sections: numbered entries in sequence; wrapped lines continue
    # the current entry only at the same type size (the copyright block on the
    # iMac's last page is smaller and is not part of note 16 or 17).
    last = 0
    current: Footnote | None = None
    note_size = None
    for page in doc.pages:
        if page.number not in endnote_pages:
            continue
        current = None  # never continue a note across a page boundary blindly
        for line in page.lines:
            if line.role != "body":
                continue
            text = line.text.strip()
            m = NOTE_START.match(text)
            if m and int(m.group(1)) == last + 1:
                last = int(m.group(1))
                current = Footnote(last, m.group(2).strip(), page.number)
                notes[f"e:{last}"] = current
                note_size = line.size
                line.role = "footnote"
            elif current is not None and note_size and abs(line.size - note_size) <= 0.4:
                current.text = _join(current.text, text)
                line.role = "footnote"

    # Page-bottom footnotes: small numbered lines in the lower part of a page
    # whose number is actually cited on that page.
    for page in doc.pages:
        if page.number in endnote_pages or page.skip:
            continue
        cited = {r for line in page.lines for r in line.refs}
        if not cited:
            continue
        current = None
        for line in page.lines:
            if line.role != "body" or line.y0 < page.height * 0.6:
                continue
            if line.size > doc.body_size - 1.0:
                current = None
                continue
            text = line.text.strip()
            m = NOTE_START.match(text)
            key = f"p{page.number}:{m.group(1)}" if m else ""
            if m and int(m.group(1)) in cited and key not in notes:
                current = Footnote(int(m.group(1)), m.group(2).strip(), page.number)
                notes[key] = current
                line.role = "footnote"
            elif current is not None:
                current.text = _join(current.text, text)
                line.role = "footnote"

    for page in doc.pages:
        for line in page.lines:
            for ref in line.refs:
                note = resolve(notes, ref, page.number)
                if note and page.number not in note.cited_on:
                    note.cited_on.append(page.number)
    return notes


def resolve(notes: dict[str, Footnote], ref: int, page: int) -> Footnote | None:
    """Page-bottom notes are numbered per page or chapter (GEP restarts them),
    so a marker resolves to the note on its own page first, then the next page
    (a note can spill over), and only then to the document's endnotes."""
    return notes.get(f"p{page}:{ref}") or notes.get(f"p{page + 1}:{ref}") or notes.get(f"e:{ref}")


def _join(a: str, b: str) -> str:
    if re.search(r"[a-z]-$", a) and re.match(r"[a-z]", b):
        return a[:-1] + b
    return f"{a} {b}"


def note_text(note: Footnote, limit: int = 450) -> str:
    text = note.text if len(note.text) <= limit else note.text[: limit - 1].rstrip() + "…"
    return f"[{note.number}] {text}"


def refs_of(lines: list[Line]) -> list[int]:
    seen: list[int] = []
    for line in lines:
        for r in line.refs:
            if r not in seen:
                seen.append(r)
    return seen
