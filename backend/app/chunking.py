"""Structure-aware chunking: section-bounded children, parent windows, headers.

Phase 1 cut every page into 1000-character windows. Here a chunk never
crosses a section boundary, is packed from whole paragraphs, and carries a
contextual header (document › section path › page) that is embedded with it.
That header is what lets "South Asia" find text that only ever says "SAR",
and what tells GEP's seven near-identical "Outlook" subsections apart.
"""

import re
from dataclasses import dataclass, field

from . import config
from .footnotes import Footnote, note_text, refs_of, resolve
from .parsing import ParsedDoc, join_lines
from .structure import Structure

SENTENCE_RE = re.compile(r"(?<=[.!?])[\"”’)]?\s+(?=[A-Z0-9“\"(])")
END_PUNCT = (".", "!", "?", ":", "”", '"', ")", "’")
TABLE_CHUNK_MAX = 2500


@dataclass
class Item:
    kind: str  # text | table | figure
    text: str
    page: int
    section: int
    refs: list[int] = field(default_factory=list)
    table: object = None
    page_end: int = 0


# --------------------------------------------------------------------------
# Reading-order items
# --------------------------------------------------------------------------


def _useful(text: str) -> bool:
    letters = sum(c.isalpha() for c in text)
    return letters >= 12 and letters / max(len(text), 1) >= 0.35


def build_items(doc: ParsedDoc) -> list[Item]:
    items: list[Item] = []
    for page in doc.pages:
        if page.skip:
            continue
        by_anchor = {id(t.anchor): t for t in page.tables}
        lines = page.lines
        buf: list = []

        def flush():
            if buf:
                text = join_lines([l.text for l in buf])
                items.append(Item("text", text, page.number, getattr(buf[0], "section", 0), refs_of(buf)))
                buf.clear()

        i = 0
        while i < len(lines):
            line = lines[i]
            section = getattr(line, "section", 0)
            if id(line) in by_anchor:
                flush()
                table = by_anchor[id(line)]
                items.append(Item("table", table.markdown(), page.number, section, table.refs, table=table))
                i += 1
                continue
            if line.role == "caption" and line.text.strip().upper().startswith(("FIGURE", "CHART", "MAP")):
                flush()
                group = [line]
                j = i + 1
                while j < len(lines) and lines[j].role == "caption":
                    group.append(lines[j])
                    j += 1
                # Panel titles, sources and notes printed small under the caption.
                while j < len(lines) and lines[j].role == "body" and lines[j].size <= doc.body_size - 0.8:
                    group.append(lines[j])
                    j += 1
                kept = [l for l in group if _useful(l.text) or l is line]
                items.append(Item("figure", join_lines([l.text for l in kept]), page.number, section, refs_of(group)))
                i = j
                continue
            if line.role != "body" and line.role != "caption":
                flush()  # headings, box markers, table cells and footnotes end a paragraph
                i += 1
                continue
            if buf and (line.block != buf[-1].block or section != getattr(buf[-1], "section", 0)):
                flush()
            buf.append(line)
            i += 1
        flush()

    # Paragraphs interrupted by a column or page break continue in the next block.
    merged: list[Item] = []
    for item in items:
        prev = merged[-1] if merged else None
        if (prev and item.kind == "text" and prev.kind == "text" and prev.section == item.section
                and not prev.text.rstrip().endswith(END_PUNCT) and item.text[:1].islower()):
            prev.text = join_lines([prev.text, item.text])
            prev.refs += [r for r in item.refs if r not in prev.refs]
            prev.page_end = item.page
            continue
        merged.append(item)
    # Drop only letterless debris (chart tick labels such as "0 2 4 6"); short
    # labelled values such as "• Production: 50%" carry the answer.
    return [i for i in merged if i.kind != "text" or sum(c.isalpha() for c in i.text) >= 3]


# --------------------------------------------------------------------------
# Chunks
# --------------------------------------------------------------------------


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]


def _short_title(title: str) -> str:
    return re.sub(r"\s+--\s+", " — ", title).strip()


class Chunker:
    def __init__(self, doc: ParsedDoc, structure: Structure, notes: dict[str, Footnote]):
        self.doc = doc
        self.sections = structure.sections
        self.notes = notes
        self.chunks: list[dict] = []
        self.parents: dict[str, dict] = {}
        self.title = _short_title(doc.title)

    # -- helpers ---------------------------------------------------------

    def section_path(self, section_id: int) -> list[str]:
        sec = self.sections[section_id] if 0 <= section_id < len(self.sections) else None
        return list(sec.path) if sec else []

    def header(self, section_id: int, page: int, kind: str) -> str:
        parts = [self.title]
        sec = self.sections[section_id] if 0 <= section_id < len(self.sections) else None
        if sec and sec.path:
            by_title = {}
            # Aliases of every section on the path, e.g. "South Asia (SAR)".
            for s in self.sections:
                if s.aliases:
                    by_title.setdefault(s.title, s.aliases)
            for title in sec.path[-3:]:
                alias = [a for a in by_title.get(title, []) if a.lower() not in title.lower()]
                parts.append(f"{title} ({', '.join(alias)})" if alias else title)
        label = self.doc.pages[page - 1].label if 0 < page <= len(self.doc.pages) else str(page)
        tag = {"table": " · table", "figure": " · figure", "visual": " · chart transcription",
               "footnote": " · footnote"}.get(kind, "")
        return " › ".join(parts) + f" · p.{label}{tag}"

    def notes_for(self, refs: list[int], page: int) -> list[str]:
        if not config.FOOTNOTES:
            return []
        out = []
        for r in refs:
            note = resolve(self.notes, r, page)
            if note:
                out.append(note_text(note))
        return out[:4]

    def add(self, kind: str, text: str, page: int, section: int, refs=None, page_end=None,
            parent_id=None, search_extra: str = "") -> dict:
        refs = refs or []
        notes = self.notes_for(refs, page)
        body = text + ("\nNotes: " + " ".join(notes) if notes else "")
        header = self.header(section, page, kind)
        chunk = {
            "id": f"{self.doc.doc_id}:c{len(self.chunks):05d}",
            "doc_id": self.doc.doc_id,
            "file_name": self.doc.file_name,
            "kind": kind,
            "page": page,
            "page_end": page_end or page,
            "page_label": self.doc.pages[page - 1].label if 0 < page <= len(self.doc.pages) else str(page),
            "section_id": section,
            "section_path": self.section_path(section),
            "parent_id": parent_id,
            "order": len(self.chunks),
            "footnotes": sorted(set(refs)) if notes else [],
            "header": header,
            "text": body,
        }
        search_body = (search_extra or body)
        chunk["search_text"] = f"{header}\n{search_body}" if config.CONTEXT_HEADERS else search_body
        self.chunks.append(chunk)
        return chunk

    # -- packing ---------------------------------------------------------

    def pack_run(self, items: list[Item]) -> None:
        """Pack one section's text items into children, and children into parents."""
        target, hard = config.CHUNK_TARGET, config.CHUNK_MAX
        pieces: list[tuple[str, Item]] = []
        for item in items:
            if len(item.text) <= hard:
                pieces.append((item.text, item))
            else:
                for s in _sentences(item.text):
                    pieces.append((s, item))

        current: list[tuple[str, Item]] = []
        children: list[dict] = []

        def emit(overlap: str):
            if not current:
                return
            text = " ".join(p for p, _ in current)
            if overlap:
                text = overlap + " " + text
            pages = [it.page for _, it in current] + [it.page_end or it.page for _, it in current]
            refs: list[int] = []
            for _, it in current:
                refs += [r for r in it.refs if r not in refs]
            chunk = self.add("text", text, min(pages), current[0][1].section, refs, max(pages))
            chunk["_body"] = " ".join(p for p, _ in current)
            children.append(chunk)

        overlap = ""
        size = 0
        for text, item in pieces:
            if current and (size + len(text) > hard or (size >= target and size + len(text) > target)):
                emit(overlap)
                last = _sentences(current[-1][0])
                overlap = last[-1] if last and len(last[-1]) <= 250 else ""
                current, size = [], 0
            current.append((text, item))
            size += len(text) + 1
        emit(overlap)
        self._parents(children)

    def _parents(self, children: list[dict]) -> None:
        group: list[dict] = []
        size = 0

        def close():
            if not group:
                return
            pid = f"{self.doc.doc_id}:p{len(self.parents):04d}"
            for c in group:
                c["parent_id"] = pid
            self.parents[pid] = {
                "id": pid,
                "doc_id": self.doc.doc_id,
                "text": "\n\n".join(c.pop("_body", c["text"]) for c in group),
                "page": min(c["page"] for c in group),
                "page_end": max(c["page_end"] for c in group),
                "section_path": group[0]["section_path"],
                "children": [c["id"] for c in group],
            }

        for child in children:
            body = len(child.get("_body", child["text"]))
            if group and size + body > config.PARENT_MAX:
                close()
                group, size = [], 0
            group.append(child)
            size += body
        close()

    def table_chunks(self, item: Item) -> None:
        table = item.table
        rows = table.rows
        batches: list[list] = []
        batch: list = []
        for row in rows:
            batch.append(row)
            if len(table.markdown(batch)) > TABLE_CHUNK_MAX and len(batch) > 1:
                batch.pop()
                batches.append(batch)
                batch = [row]
        if batch:
            batches.append(batch)
        for part in batches:
            sentences = "\n".join(table.row_sentences(part))
            extra = "\n".join(x for x in (table.caption, table.subtitle, sentences) if x)
            self.add("table", table.markdown(part), item.page, item.section, item.refs, search_extra=extra)


def make_chunks(doc: ParsedDoc, structure: Structure, notes: dict[str, Footnote],
                visuals: dict[int, list[str]] | None = None) -> tuple[list[dict], dict[str, dict]]:
    chunker = Chunker(doc, structure, notes)
    items = build_items(doc)

    page_section: dict[int, int] = {}
    run: list[Item] = []
    for item in items:
        page_section.setdefault(item.page, item.section)
        if item.kind == "text":
            if run and run[-1].section != item.section:
                chunker.pack_run(run)
                run = []
            run.append(item)
            continue
        if run:
            chunker.pack_run(run)
            run = []
        if item.kind == "table":
            chunker.table_chunks(item)
        else:
            chunker.add(item.kind, item.text, item.page, item.section, item.refs)
    if run:
        chunker.pack_run(run)

    # Chart transcriptions sit in the section of the page they describe.
    last = 0
    for number in range(1, doc.page_count + 1):
        last = page_section.get(number, last)
        for text in (visuals or {}).get(number, []):
            if text.strip():
                chunker.add("visual", f"Chart/infographic transcribed from page {doc.pages[number - 1].label}: {text}",
                            number, last)

    # Each footnote is also searchable on its own.
    if config.FOOTNOTES:
        for note in notes.values():
            cited = ", ".join(doc.pages[p - 1].label for p in sorted(note.cited_on)[:12])
            text = f"Footnote {note.number}: {note.text}" + (f"\n(Cited on page {cited}.)" if cited else "")
            chunker.add("footnote", text, note.page, page_section.get(note.page, last))

    for chunk in chunker.chunks:
        chunk.pop("_body", None)
    return chunker.chunks, chunker.parents
