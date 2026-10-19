"""Numeric faithfulness: every figure in an answer must appear in its sources.

Wrong numbers are the most damaging hallucination in these documents (a
region's growth rate, a product's footprint), and the cheapest to check: no
model call is needed, only a comparison against the context the model saw.
Equivalent spellings are accepted: "874.5 million" vs "$874,466,000", "45%" vs
"45 percent", a rounded "375 million" vs "$375.7 million".
"""

import re

NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?!\w)")
SOURCE_RE = re.compile(r"\[(?:SOURCE|EXCERPT|REF)S?\s*[\d,\s]+\]", re.IGNORECASE)
LIST_MARK_RE = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
SCALES = (1.0, 1e3, 1e6, 1e9)


def _numbers(text: str) -> list[tuple[str, float, int]]:
    out = []
    for m in NUM_RE.finditer(text):
        whole, frac = m.group(1), m.group(2)
        raw = m.group(0)
        value = float(whole.replace(",", "") + ("." + frac if frac else ""))
        out.append((raw, value, len(frac) if frac else 0))
    return out


YEAR_TOKEN_RE = re.compile(r"(?<!\d)((?:19|20)\d\d)(?=[efp]\b|/\d\d)")


def _context_values(context: str) -> set[float]:
    values = {v for _, v, _ in _numbers(context)}
    # Forecast tables label years "2028f", "2025e", "2025/26"; the year itself counts.
    values.update(float(y) for y in YEAR_TOKEN_RE.findall(context))
    return values


def _supported(value: float, decimals: int, ctx: set[float], raw: str, answer: str, pos: int) -> bool:
    if value in ctx:
        return True
    # "2027–28": the second half is a year abbreviation, checked via its partner.
    before = answer[max(0, pos - 6):pos]
    if value < 100 and re.search(r"\d{4}\s?[–-]$", before):
        return True
    tol = 0.5 * 10 ** (-decimals) if decimals else 0.5
    for c in ctx:
        for scale in SCALES:
            # Answer in millions, context in units (or the reverse, e.g. tables
            # "in thousands").
            if abs(value * scale - c) <= tol * scale or abs(value - c * scale) <= tol:
                return True
    return False


def unsupported_numbers(answer: str, context: str) -> list[str]:
    clean = SOURCE_RE.sub(" ", answer)
    clean = LIST_MARK_RE.sub(" ", clean)
    ctx = _context_values(context)
    bad: list[str] = []
    for m in NUM_RE.finditer(clean):
        raw = m.group(0)
        whole, frac = m.group(1), m.group(2)
        value = float(whole.replace(",", "") + ("." + frac if frac else ""))
        if not _supported(value, len(frac) if frac else 0, ctx, raw, clean, m.start()):
            if raw not in bad:
                bad.append(raw)
    return bad
