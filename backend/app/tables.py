"""Table reconstruction from text coordinates, plus key/value line merging.

PyMuPDF's find_tables() was tried on all three test documents and misread
them: GEP Table 2.10 lost its year headers and the iMac power table came back
as one-row fragments. Both documents do place every cell as its own line on a
shared baseline, so rows are rebuilt by clustering baselines and columns by
aligning cells under the header row.
"""

import re
from dataclasses import dataclass, field

from .parsing import Line, Page

NUM_CELL = re.compile(
    r"^(?:[−–-]?\(?[$€£]?\s?\d[\d,]*(?:\.\d+)?\)?\s?(?:%|W|kW|kWh|kg|g|x|pp|bn|m|M|B)?"
    r"|\.\.|n\.?a\.?|[<>≤≥]\s?\d+(?:\.\d+)?%?|[—–-])$",
    re.IGNORECASE,
)
YEAR_CELL = re.compile(r"^(?:(?:19|20)\d\d(?:/\d\d)?[efp]?|\d\dQ\d[efp]?)$")
CAPTION_RE = re.compile(r"^(?:TABLE|Table)\s+[A-Z]?\d+(?:\.\d+)*\b")
CURRENCY_ONLY = re.compile(r"^[$€£]$")
TRAILING_NUMS = re.compile(
    r"^(.*?[A-Za-z)\]].*?)((?:\s+(?:[−–-]?\(?\$?\d[\d,]*(?:\.\d+)?\)?\s?(?:%|W|kg)?|\.\.)){2,})$"
)


@dataclass
class Cell:
    x0: float
    x1: float
    text: str

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class Row:
    y: float
    cells: list[Cell]
    lines: list[Line]

    @property
    def numeric(self) -> list[Cell]:
        return [c for c in self.cells if _is_value(c.text)]


def _is_value(text: str) -> bool:
    return bool(NUM_CELL.match(text)) and not YEAR_CELL.match(text)


@dataclass
class Table:
    page: int
    caption: str
    subtitle: str
    label_header: str
    columns: list[str]
    rows: list[tuple[str, str, list[str]]]  # (group, label, values)
    notes: str
    anchor: Line  # first line of the table in reading order
    refs: list[int] = field(default_factory=list)

    @property
    def title(self) -> str:
        return (self.caption or self.label_header or "Table")[:120]

    def markdown(self, rows: list | None = None) -> str:
        rows = self.rows if rows is None else rows
        head = [self.label_header or "Item"] + self.columns
        out = []
        if self.caption:
            out.append(self.caption)
        if self.subtitle:
            out.append(self.subtitle)
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "---|" * len(head))
        group = None
        for g, label, values in rows:
            if g and g != group:
                out.append("| **" + g + "** |" + " |" * len(self.columns))
                group = g
            padded = (values + [""] * len(self.columns))[: len(self.columns)]
            out.append("| " + " | ".join([label] + padded) + " |")
        if self.notes:
            out.append("Notes: " + self.notes)
        return "\n".join(out)

    def row_sentences(self, rows: list | None = None) -> list[str]:
        """One retrieval-friendly sentence per row: header names travel with the values."""
        rows = self.rows if rows is None else rows
        unit = f" ({self.subtitle.strip('()')})" if self.subtitle else ""
        out = []
        for group, label, values in rows:
            pairs = ", ".join(f"{c}: {v}" for c, v in zip(self.columns, values) if v)
            where = f"{group} — " if group else ""
            out.append(f"{self.title}{unit} — {where}{label}: {pairs}.")
        return out


# --------------------------------------------------------------------------
# Row building
# --------------------------------------------------------------------------


def _cells_for(line: Line) -> list[Cell]:
    """Split a line into cells; a label followed by several numbers is several cells."""
    text = line.text.strip()
    if CURRENCY_ONLY.match(text):
        return []
    m = TRAILING_NUMS.match(text)
    if not m:
        return [Cell(line.x0, line.x1, text)]
    label, rest = m.group(1).strip(), m.group(2).split()
    # Approximate x positions by character offset; only the order matters here.
    per_char = line.width / max(len(text), 1)
    cells = [Cell(line.x0, line.x0 + per_char * len(label), label)]
    pos = len(label)
    for token in rest:
        idx = text.find(token, pos)
        pos = idx + len(token)
        cells.append(Cell(line.x0 + per_char * idx, line.x0 + per_char * pos, token))
    return cells


def _rows(lines: list[Line]) -> list[Row]:
    rows: list[Row] = []
    for line in sorted(lines, key=lambda l: (l.yc, l.x0)):
        tol = 0.45 * max(line.size, 4)
        if rows and abs(rows[-1].y - line.yc) <= tol:
            rows[-1].lines.append(line)
        else:
            rows.append(Row(y=line.yc, cells=[], lines=[line]))
    for row in rows:
        row.lines.sort(key=lambda l: l.x0)
        for line in row.lines:
            row.cells.extend(_cells_for(line))
        row.cells.sort(key=lambda c: c.x0)
    return rows


def _is_data_row(row: Row) -> bool:
    nums = len(row.numeric)
    others = len(row.cells) - nums
    first = row.cells[0] if row.cells else None
    has_label = first is not None and not NUM_CELL.match(first.text) and not YEAR_CELL.match(first.text)
    return nums >= 2 and others <= 2 and has_label


def _is_header_row(row: Row) -> bool:
    if len(row.cells) < 2 or len(row.cells) > 16:
        return False
    if any(len(c.text) > 40 for c in row.cells):
        return False
    years = sum(bool(YEAR_CELL.match(c.text)) for c in row.cells)
    return not row.numeric and (years >= 2 or all(len(c.text) <= 25 for c in row.cells))


def _is_group_row(row: Row) -> bool:
    if len(row.cells) != 1:
        return False
    text = row.cells[0].text.strip()
    return len(text) <= 60 and not NUM_CELL.match(text) and not CAPTION_RE.match(text)


def _is_annotation_row(row: Row) -> bool:
    """A parenthesised note inside a table, e.g. '(Average including economies...)'."""
    return len(row.cells) == 1 and row.cells[0].text.strip().startswith("(")


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def extract_tables(page: Page, body_size: float) -> list[Table]:
    """Find tables on a page, mark their lines role='table', return them."""
    candidates = [l for l in page.lines if l.role == "body"]
    if len(candidates) < 8:
        return []
    rows = _rows(candidates)
    tables: list[Table] = []
    i = 0
    while i < len(rows):
        if not _is_data_row(rows[i]):
            i += 1
            continue
        region = _grow_region(rows, i, body_size)
        if sum(_is_data_row(r) for r in region) >= 3:
            table = _build(page, rows, region, body_size, tables[-1] if tables else None)
            if table:
                tables.append(table)
        i = rows.index(region[-1]) + 1
    page.tables.extend(tables)
    return tables


def _grow_region(rows: list[Row], start: int, body_size: float) -> list[Row]:
    pitch = 3.2 * body_size
    region = [rows[start]]
    j = start + 1
    while j < len(rows):
        row = rows[j]
        if row.y - region[-1].y > pitch:
            break
        if _is_data_row(row) or _is_header_row(row) or _is_group_row(row) or _is_annotation_row(row):
            region.append(row)
            j += 1
            continue
        break
    # Trailing group/header rows belong to whatever follows, not to this table.
    while region and not _is_data_row(region[-1]):
        region.pop()
    return region


def _value_cells(row: Row, label_right: float) -> list[Cell]:
    return [c for c in row.cells if c.xc > label_right]


def _build(page: Page, rows: list[Row], region: list[Row], body_size: float,
           previous: "Table | None") -> "Table | None":
    first_idx = rows.index(region[0])
    data = [r for r in region if _is_data_row(r)]
    counts = [len(r.numeric) for r in data]
    n_cols = max(set(counts), key=counts.count)
    label_right = max(r.cells[0].x1 for r in data)

    # Walk upwards: group labels, then the header row, then super-headers.
    used: list[Line] = []
    groups_above: list[Row] = []
    header: Row | None = None
    super_rows: list[Row] = []
    k = first_idx - 1
    while k >= 0 and region[0].y - rows[k].y <= 9 * body_size:
        row = rows[k]
        if _is_header_row(row) and len(_value_cells(row, label_right)) >= min(2, n_cols):
            header = row
            break
        if _is_annotation_row(row):
            used.extend(row.lines)
            k -= 1
            continue
        if _is_group_row(row) and not CAPTION_RE.match(row.cells[0].text):
            groups_above.insert(0, row)
            k -= 1
            continue
        break
    if header is not None:
        k -= 1
        while k >= 0 and header.y - rows[k].y <= 3.5 * body_size and len(super_rows) < 4:
            row = rows[k]
            if _is_data_row(row):
                break
            caption_cells = [c for c in row.cells if CAPTION_RE.match(c.text)]
            if caption_cells:
                # Header fragments can share the caption's baseline ("TABLE 2.9 ..."
                # on the left, "Percentage-point" over the last two columns).
                cap_x1 = max(c.x1 for c in caption_cells)
                rest = [c for c in row.cells if c not in caption_cells and c.x0 > cap_x1 + 20]
                if rest:
                    lines = [l for l in row.lines if l.x0 > cap_x1 + 20]
                    super_rows.insert(0, Row(y=row.y, cells=rest, lines=lines))
                break
            super_rows.insert(0, row)
            k -= 1

    columns: list[str] = []
    label_header = ""
    subtitle_parts: list[str] = []
    title_parts: list[str] = []
    if header is not None:
        vcells = _value_cells(header, label_right)[-n_cols:]
        columns = [c.text for c in vcells]
        label_header = " ".join(c.text for c in header.cells if c.xc <= label_right)
        span_lo = min(c.x0 for c in vcells) if vcells else 0
        span_hi = max(c.x1 for c in vcells) if vcells else 0
        prefixes = [""] * len(columns)
        for sr in super_rows:
            for sc in sr.cells:
                text = sc.text.strip()
                if text.startswith("("):
                    subtitle_parts.append(text)
                    continue
                if sc.x1 <= span_lo:  # sits over the label column
                    label_header = (label_header + " " + text).strip()
                    continue
                centre = (span_lo + span_hi) / 2
                if abs(sc.xc - centre) <= 0.15 * max(span_hi - span_lo, 1):
                    title_parts.append(text)  # centred over all columns: a title
                    continue
                for idx, col in enumerate(vcells):
                    if sc.x0 - 8 <= col.xc <= sc.x1 + 8:
                        prefixes[idx] = (prefixes[idx] + " " + text).strip()
        columns = [f"{p} {c}".strip() for p, c in zip(prefixes, columns)]
        used.extend(header.lines)
        for sr in super_rows:
            used.extend(sr.lines)
    elif previous is not None and len(previous.columns) == n_cols:
        columns = list(previous.columns)  # continuation of the table above
        label_header = previous.label_header
    if len(columns) < n_cols:
        columns = columns + [f"Col {i + 1}" for i in range(len(columns), n_cols)]

    group = " — ".join(r.cells[0].text.strip() for r in groups_above[-1:])
    for r in groups_above:
        used.extend(r.lines)

    body: list[tuple[str, str, list[str]]] = []
    refs: list[int] = []
    for row in region:
        used.extend(row.lines)
        for line in row.lines:
            refs.extend(line.refs)
        if _is_header_row(row) and not _is_data_row(row):
            continue  # repeated sub-header (e.g. GEP's fiscal-year basis years)
        if _is_annotation_row(row):
            continue
        if _is_group_row(row):
            group = row.cells[0].text.strip()
            continue
        label_cells = [c for c in row.cells if not _is_value(c.text)]
        values = [c.text for c in row.cells if _is_value(c.text)]
        label = " ".join(c.text.strip() for c in label_cells)
        label = re.sub(r"(\s+\d{1,2})+$", "", label)  # trailing footnote markers
        body.append((group, label, values[-n_cols:]))

    caption, subtitle, cap_lines = _caption(page, region, header, super_rows, body_size, set(map(id, used)))
    used.extend(cap_lines)
    subtitle = " ".join([subtitle] + subtitle_parts).strip()
    if not caption and title_parts:
        caption = " ".join(title_parts)
    elif title_parts:
        subtitle = (subtitle + " " + " ".join(title_parts)).strip()
    if not caption and previous is not None and header is None:
        caption = previous.caption + " (continued)" if previous.caption else ""
    notes, note_lines = _notes(page, region, body_size, set(map(id, used)))
    used.extend(note_lines)

    for line in used:
        line.role = "table"
    order = {id(l): i for i, l in enumerate(page.lines)}
    anchor = min(used, key=lambda l: order.get(id(l), 10**9))
    return Table(
        page=page.number, caption=caption, subtitle=subtitle, label_header=label_header,
        columns=columns, rows=body, notes=notes, anchor=anchor, refs=sorted(set(refs)),
    )


def _caption(page: Page, region: list[Row], header: Row | None, super_rows: list[Row],
             body_size: float, used: set[int]):
    top_row = super_rows[0] if super_rows else header if header else region[0]
    top = min(l.y0 for l in top_row.lines)
    above = [
        l for l in page.lines
        if l.role == "body" and id(l) not in used and l.y0 <= top + 2 and top - l.y1 < 140
    ]
    above.sort(key=lambda l: l.y0)
    for idx, line in enumerate(above):
        if CAPTION_RE.match(line.text.strip()):
            group = [line]
            for nxt in above[idx + 1 :]:
                if nxt.y0 - group[-1].y1 > 1.8 * body_size or _is_value(nxt.text.strip()):
                    break
                group.append(nxt)
            caption = line.text.strip()
            subtitle = " ".join(l.text.strip() for l in group[1:] if l.text.strip().startswith("("))
            rest = [l.text.strip() for l in group[1:] if not l.text.strip().startswith("(")]
            if rest and not caption.endswith(")"):
                caption = caption + " " + " ".join(r for r in rest if len(r) < 90)
            return caption, subtitle, group
    return "", "", []


def _notes(page: Page, region: list[Row], body_size: float, used: set[int]):
    bottom = max(l.y1 for l in region[-1].lines)
    below = sorted(
        (l for l in page.lines
         if l.role == "body" and id(l) not in used and l.y0 >= bottom - 1 and l.y0 - bottom < 10 * body_size),
        key=lambda l: l.y0,
    )
    picked: list[Line] = []
    for line in below:
        text = line.text.strip()
        starts = re.match(r"^(Note|Notes|Source|Sources)\b", text)
        small = line.size <= body_size - 1.0
        if not picked and not starts:
            break
        if picked and (line.y0 - picked[-1].y1 > 1.5 * line.size or not small):
            break
        picked.append(line)
    return " ".join(l.text.strip() for l in picked), picked


# --------------------------------------------------------------------------
# Key / value pairs
# --------------------------------------------------------------------------

VALUE_RE = re.compile(r"\d")


def merge_key_values(page: Page) -> None:
    """Join 'label ......... value' layouts that extract as two separate lines.

    GPO's "By the Numbers" page and the iMac footprint table put each label and
    its value on the same baseline in different columns. Joined, the value is
    no longer separated from what it measures.
    """
    lines = [l for l in page.lines if l.role == "body"]
    # Lines of a multi-line paragraph block are prose; a two-column page has
    # prose lines side by side on the same baseline and they must stay apart.
    # A block qualifies as a label column when its lines are short and
    # sentence-free, and as a value column when its lines are mostly values.
    blocks: dict[int, list[Line]] = {}
    for l in page.lines:
        blocks.setdefault(l.block, []).append(l)

    def label_block(b: int) -> bool:
        ls = blocks.get(b, [])
        if len(ls) <= 3:
            return True
        short = sum(len(l.text) <= 60 and not l.text.rstrip().endswith(".") for l in ls)
        return short >= 0.8 * len(ls)

    def value_block(b: int) -> bool:
        ls = blocks.get(b, [])
        if len(ls) <= 3:
            return True
        vals = sum(bool(VALUE_RE.search(l.text)) and len(l.text.strip()) <= 28 for l in ls)
        return vals >= 0.7 * len(ls)

    consumed: set[int] = set()
    for left in lines:
        if id(left) in consumed or len(left.text) > 80 or NUM_CELL.match(left.text.strip()):
            continue
        if not label_block(left.block):
            continue
        best = None
        for right in lines:
            if right is left or id(right) in consumed:
                continue
            if not value_block(right.block):
                continue
            # The same PDF text block means the same visual row (GPO prints
            # "GPO staff ........ 1,737" across the full page width).
            max_gap = page.width if right.block == left.block else 0.45 * page.width
            if right.x0 <= left.x1 or right.x0 - left.x1 > max_gap:
                continue
            overlap = min(left.y1, right.y1) - max(left.y0, right.y0)
            if overlap < 0.5 * min(left.y1 - left.y0, right.y1 - right.y0):
                continue
            text = right.text.strip()
            if not VALUE_RE.search(text) or len(text) > 28:
                continue
            if best is None or right.x0 < best.x0:
                best = right
        if best is not None:
            sep = " " if left.text.rstrip().endswith((":", "-", "–")) else ": "
            left.text = left.text.rstrip() + sep + best.text.strip()
            left.x1 = best.x1
            left.refs.extend(best.refs)
            consumed.add(id(best))
    if consumed:
        page.lines = [l for l in page.lines if id(l) not in consumed]
