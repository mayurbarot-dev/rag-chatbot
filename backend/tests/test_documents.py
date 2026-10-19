"""Document-shape tests against the three Phase 2 PDFs. No API key needed.

Each assertion pins down one finding from the document analysis in
plans/phase 2/IMPLEMENTATION-PLAN.md section 0, so a regression in parsing,
tables, footnotes or structure shows up here before it shows up in answers.
"""
import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
PROJECT = BACKEND.parent
TMP = Path(tempfile.mkdtemp())
os.environ["RAG_DATA_DIR"] = str(TMP / "data")
os.environ["RAG_CACHE_DIR"] = str(TMP / "cache")
os.environ["VISION_MODE"] = "off"
sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402

from app.chunking import make_chunks  # noqa: E402
from app.footnotes import extract_footnotes  # noqa: E402
from app.grounding import unsupported_numbers  # noqa: E402
from app.parsing import join_lines, parse_pdf, repair_text  # noqa: E402
from app.retrieval import analyse  # noqa: E402
from app.store import Store, tokenize  # noqa: E402
from app.structure import build_structure  # noqa: E402
from app.tables import extract_tables, merge_key_values  # noqa: E402

DOCS = PROJECT / "plans" / "phase 2"
ok = fail = 0


def check(label, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label} {extra}")


def prepare(name):
    doc = parse_pdf((DOCS / name).read_bytes(), name)
    for page in doc.pages:
        if not page.skip:
            extract_tables(page, doc.body_size)
            merge_key_values(page)
    notes = extract_footnotes(doc)
    structure = build_structure(doc)
    chunks, parents = make_chunks(doc, structure, notes, {})
    return doc, notes, structure, chunks, parents


def page_text(doc, n):
    return "\n".join(line.text for line in doc.pages[n - 1].lines)


print("\n== Text repair ==")
check("Th ligature repaired", repair_text("Te conflict and Tese findings", True) == "The conflict and These findings")
check("repair is off for normal docs", repair_text("Te Deum", False) == "Te Deum")
check("NFKC fixes fi/fl ligatures", repair_text("conﬂict ﬁscal") == "conflict fiscal")
check("de-hyphenation", join_lines(["confi-", "dence and low-", "Carbon"]) == "confidence and low- Carbon")

gpo, gpo_notes, gpo_struct, gpo_chunks, _ = prepare("GPO-Anual-report.pdf")
gep, gep_notes, gep_struct, gep_chunks, gep_parents = prepare("GEP-Jun-2026.pdf")
imac, imac_notes, imac_struct, imac_chunks, _ = prepare("iMac_PER_Oct2024.pdf")

print("\n== GPO: narrative report ==")
check("letter-spaced running footer removed", "G O V E R N M E N T" not in page_text(gpo, 7))
check("drop cap rejoined", "The U.S. Government Publishing Office (GPO)," in page_text(gpo, 7))
check("printed page label p5 -> 3", gpo.pages[4].label == "3", gpo.pages[4].label)
check("cover page label c1", gpo.pages[0].label == "c1", gpo.pages[0].label)
check("apostrophes intact", "GPO’s" in page_text(gpo, 5) and "�" not in page_text(gpo, 5))
check("'By the Numbers' label joined to value", "GPO staff: 1,737" in page_text(gpo, 14), page_text(gpo, 14)[:200])
check("166 libraries fact is in a chunk", any("166 depository libraries" in c["text"] for c in gpo_chunks))
balance = next((t for p in gpo.pages for t in p.tables if p.number == 58), None)
check("balance sheet rebuilt with year columns", balance is not None and balance.columns == ["2018", "2017"],
      balance.columns if balance else None)
check("narrative pages get a section", any(
    c["page"] == 5 and c["section_path"] for c in gpo_chunks))
check("profile is narrative", gpo_struct.profile == "narrative", gpo_struct.profile)

print("\n== GEP: long structured report ==")
check("Th ligature repair applied", gep.th_repair and "The global economy is facing" in page_text(gep, 19))
check("contents pages skipped", all(gep.pages[n - 1].skip == "toc" for n in (9, 10, 11, 12, 13)))
check("abbreviations page parsed to glossary", gep.glossary.get("SAR") == "South Asia", gep.glossary.get("SAR"))
check("printed label p96 -> 74", gep.pages[95].label == "74", gep.pages[95].label)
check("roman front-matter label p19 -> xvii", gep.pages[18].label == "xvii", gep.pages[18].label)
south = [s for s in gep_struct.sections if s.title == "South Asia"]
check("South Asia snapped from outline p95 to its heading on p96", south and south[0].page == 96,
      [(s.title, s.page) for s in south])
check("South Asia nested under Chapter 2", south and south[0].path[0].startswith("Chapter 2"), south[0].path if south else None)
check("South Asia aliased to SAR", south and "SAR" in south[0].aliases)
sar = next((c for c in gep_chunks if "Growth in SAR is expected to soften to 6.3" in c["text"]), None)
check("SAR outlook chunk is in South Asia › Outlook",
      sar is not None and sar["section_path"][-2:] == ["South Asia", "Outlook"], sar["section_path"] if sar else None)
check("SAR chunk header carries the alias", sar is not None and "South Asia (SAR)" in sar["header"])
check("SAR chunk has a parent window", sar is not None and sar["parent_id"] in gep_parents)
box = [c for c in gep_chunks if c["page"] == 31 and c["kind"] == "text"]
check("Box 1.1 pages belong to the box", box and all(any(p.startswith("Box 1.1") for p in c["section_path"]) for c in box),
      [c["section_path"] for c in box][:2])
after_box = [c for c in gep_chunks if c["page"] == 36 and c["kind"] == "text"]
check("text after Box 1.1 returns to Chapter 1", after_box and not any(
    p.startswith("Box") for c in after_box for p in c["section_path"]))
exec_chunks = [c for c in gep_chunks if c["page"] == 19]
check("executive summary section", exec_chunks and all(c["section_path"][:1] == ["Executive Summary"] for c in exec_chunks))
t210 = next((t for p in gep.pages if p.number == 98 for t in p.tables if t.caption.startswith("TABLE 2.10")), None)
check("Table 2.10 rebuilt", t210 is not None)
if t210:
    maldives = next((r for r in t210.rows if r[1] == "Maldives"), None)
    check("Table 2.10 2026f column", t210.columns[3] == "2026f", t210.columns)
    check("Maldives 2026f = 0.7", maldives is not None and maldives[2][3] == "0.7", maldives)
    check("difference columns keep their super-header",
          t210.columns[-1].startswith("Percentage-point differences"), t210.columns[-1])
check("table chunks searchable by row sentence",
      any(c["kind"] == "table" and "Maldives: 2023: 4.9" in c["search_text"] for c in gep_chunks))
check("profile is long-structured", gep_struct.profile == "long-structured", gep_struct.profile)

print("\n== iMac: visual report ==")
check("12 pages (not 11)", imac.page_count == 12)
energy = next((l for l in imac.pages[6].lines if "58 percent less energy" in l.text), None)
check("superscript marker removed from text", energy is not None and "STAR.18" not in energy.text)
check("superscript marker recorded as ref 18", energy is not None and 18 in energy.refs, energy.refs if energy else None)
check("19 endnotes parsed", len([k for k in imac_notes if k.startswith("e:")]) == 19, sorted(imac_notes))
check("note 18 mentions the 143W adapter", "143W" in imac_notes["e:18"].text)
check("note 18 cited on p7", 7 in imac_notes["e:18"].cited_on, imac_notes["e:18"].cited_on)
check("copyright block not glued onto a note", "©" not in imac_notes["e:16"].text and "©" not in imac_notes["e:17"].text)
ec = next((c for c in imac_chunks if "58 percent less energy" in c["text"] and c["kind"] == "text"), None)
check("energy chunk carries footnote 18", ec is not None and "[18]" in ec["text"] and 18 in ec["footnotes"])
power = next((t for p in imac.pages for t in p.tables), None)
check("power table rebuilt", power is not None and power.columns == ["100V", "115V", "230V"], power.columns if power else None)
sleep = next((r for r in power.rows if r[1] == "Sleep"), None) if power else None
check("sleep at 115V = 0.57W", sleep is not None and sleep[2][1] == "0.57W", sleep)
check("footprint split kept", any("Production: 50%" in c["text"] and "Product use: 45%" in c["text"] for c in imac_chunks))
check("spacing restored around removed markers", "suppliers.By" not in page_text(imac, 4))
check("footnote chunks exist", sum(c["kind"] == "footnote" for c in imac_chunks) == 19)
check("profile is visual-dense", imac_struct.profile == "visual-dense", imac_struct.profile)

print("\n== Grounding check ==")
check("exact figure supported", unsupported_numbers("Growth is 6.3 percent [SOURCE 1].", "soften to 6.3 percent") == [])
check("scaled figure supported", unsupported_numbers("Revenues were $874.5 million.", "revenues $874,466,000") == [])
check("thousands table supported", unsupported_numbers("Revenue was $874,466,000.", "Total operating revenues 874,466") == [])
check("year range shorthand supported", unsupported_numbers("in 2027–28", "pick up in 2027–28") == [])
check("forecast-year labels count as years", unsupported_numbers("7.1 percent in 2028", "2027f 2028f 7.1") == [])
check("invented figure caught", unsupported_numbers("Growth is 7.5%.", "Growth is 6.3%.") == ["7.5"])
check("source markers ignored", unsupported_numbers("See [SOURCE 12].", "nothing") == [])

print("\n== Query analysis ==")
store = Store(TMP / "qa-index")
for doc, struct, chunks in ((gpo, gpo_struct, gpo_chunks), (gep, gep_struct, gep_chunks), (imac, imac_struct, imac_chunks)):
    record = {"doc_id": doc.doc_id, "file_name": doc.file_name, "title": doc.title, "glossary": doc.glossary,
              "aliases": {s.title: s.aliases for s in struct.sections if s.aliases}}
    store.add_document(record, chunks, np.zeros((len(chunks), 4), dtype=np.float32), {}, "test")
plan = analyse(store, "What is the growth outlook for South Asia in this report?")
check("routes South Asia to GEP", plan.doc_ids == {gep.doc_id}, plan.doc_ids)
check("boosts the South Asia section", "South Asia" in plan.soft_sections, plan.soft_sections)
check("expands South Asia -> SAR for keywords", "sar" in plan.bm25_extra, plan.bm25_extra)
plan = analyse(store, "What does Chapter 3 say about sovereign debt in developing economies?")
check("Chapter 3 becomes a hard filter", plan.hard_reason.startswith("Chapter 3"), plan.hard_reason)
check("summary intent detected", plan.summary)
plan = analyse(store, "According to the executive summary, what are the main risks to the global outlook?")
check("executive summary hard filter", "Executive Summary" in plan.hard_reason, plan.hard_reason)
plan = analyse(store, "According to Table 2.10, what is the 2026 growth forecast for Maldives?")
check("table reference hard filter", "table 2.10" in plan.hard_reason, plan.hard_reason)
plan = analyse(store, "What material is the iMac stand made from?")
check("routes iMac questions to the iMac report", plan.doc_ids == {imac.doc_id}, plan.doc_ids)
plan = analyse(store, "How many depository libraries did GPO staff visit?")
check("routes GPO questions to the GPO report", plan.doc_ids == {gpo.doc_id}, plan.doc_ids)
from app.retrieval import split_parts  # noqa: E402
check("two-part question split", split_parts("What new digital system replaced FDsys, and when was it fully deployed?")
      == ["What new digital system replaced FDsys?", "when was it fully deployed?"])
check("single question not split", split_parts("What is the growth outlook for South Asia in this report?") == [])
check("tokenizer keeps decimals and percents", {"6.3", "45%", "802,834"} <= set(tokenize("6.3 45% 802,834")))

print(f"\n{'=' * 46}\n  {ok} passed, {fail} failed\n{'=' * 46}")
sys.exit(1 if fail else 0)
