"""Evaluate the RAG pipeline against a fixed question set.

Runs in-process against an isolated data directory (backend/eval/.work by
default), so the app's own index and memory are never touched.

    python eval/run_eval.py                       # index (if needed) + evaluate
    python eval/run_eval.py --reindex             # force re-indexing
    python eval/run_eval.py --label no-rerank --set RERANK=off
    python eval/run_eval.py --only W1,W2 --verbose

Each --set KEY=VALUE overrides a config value for that run (ablation). The
report is written to eval/results/<timestamp>-<label>.md and .json.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

REFUSAL_RE = re.compile(
    r"couldn'?t find|could not find|can'?t find|cannot find|not (?:mentioned|covered|"
    r"included|available|provided|specified|stated|addressed|discussed)|"
    r"no (?:information|mention|data|details)|does(?:n'?t| not) (?:say|mention|contain|"
    r"provide|include|specify|state|discuss|cover|address)|isn'?t (?:there|in)|"
    r"do(?:es)?n'?t (?:have|contain)|not in the (?:uploaded |provided )?(?:documents?|sources?)",
    re.IGNORECASE,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="run")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--only", default="", help="Comma-separated question ids")
    p.add_argument("--reindex", action="store_true")
    p.add_argument("--work-dir", default=str(EVAL_DIR / ".work"))
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--pipeline",
        choices=["phase2", "phase1"],
        default="phase2",
        help="phase1 runs the frozen Phase 1 engine (eval/phase1_rag.py) as the baseline",
    )
    return p.parse_args()


def build_phase1_engine(work: Path):
    """The Phase 1 engine, with only its model access swapped for the shared,
    rate-limited, cached one, so baseline and Phase 2 differ in pipeline alone."""
    sys.path.insert(0, str(EVAL_DIR))
    import phase1_rag  # noqa: E402

    from app.llm import LLM  # noqa: E402

    llm = LLM()

    class Phase1(phase1_rag.RAG):
        def __init__(self):
            self.client = None
            self.store = phase1_rag.Store(path=work / "data" / "phase1_index.json")
            self.memory = phase1_rag.Memory(path=work / "data" / "phase1_memory.json")

        def _embed(self, texts, task_type):
            return [v.tolist() for v in llm._embed_cached(texts, task_type)]

        def _generate(self, prompt):
            return llm.generate(prompt)

    return Phase1()


def apply_overrides(pairs):
    # Overrides must be in the environment before app.config is imported.
    for pair in pairs:
        key, _, value = pair.partition("=")
        os.environ[key.strip()] = value.strip()


def is_refusal(result) -> bool:
    return (not result.get("grounded")) or bool(REFUSAL_RE.search(result.get("answer", "")))


def grade(q, result):
    answer = result.get("answer", "")
    pages = {s.get("page") for s in result.get("sources", [])}
    for s in result.get("sources", []):
        if s.get("page_end"):
            pages.update(range(s["page"], s["page_end"] + 1))
    expected = set(q.get("expected_pages", []))

    missing = [p for p in q.get("must_contain", []) if not re.search(p, answer, re.I)]
    forbidden = [p for p in q.get("must_not_contain", []) if re.search(p, answer, re.I)]

    if q["type"] == "negative":
        passed = is_refusal(result)
    else:
        passed = not missing and not forbidden and not (
            result.get("grounded") is False and REFUSAL_RE.search(answer)
        )

    hit = bool(expected & pages) if expected else None
    mrr = None
    if expected:
        mrr = 0.0
        for rank, s in enumerate(result.get("sources", []), start=1):
            span = set(range(s.get("page", 0), (s.get("page_end") or s.get("page", 0)) + 1))
            if span & expected:
                mrr = 1.0 / rank
                break

    sec_precision = None
    want = q.get("expected_section")
    if want and result.get("sources"):
        paths = [" › ".join(s.get("section_path") or []) for s in result["sources"]]
        if any(paths):
            sec_precision = sum(want.lower() in p.lower() for p in paths) / len(paths)
        else:
            # Phase 1 sources carry no section; fall back to page membership.
            sec_precision = sum(s.get("page") in expected for s in result["sources"]) / len(
                result["sources"]
            )

    return {
        "passed": passed,
        # Strict: the right answer AND an expected page actually in the context,
        # so an answer that happens to match from the wrong section does not count.
        "strict": passed and (hit is not False),
        "missing": missing,
        "forbidden": forbidden,
        "hit": hit,
        "mrr": mrr,
        "section_precision": sec_precision,
        "unsupported": (result.get("grounding") or {}).get("unsupported", []),
    }


def main():
    args = parse_args()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    os.environ["RAG_DATA_DIR"] = str(work / "data")
    apply_overrides(args.set)

    from app import config, rag  # noqa: E402  (after env overrides)

    spec = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    questions = spec["questions"]
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        questions = [q for q in questions if q["id"] in wanted]

    engine = build_phase1_engine(work) if args.pipeline == "phase1" else rag.RAG()
    if args.reindex:
        engine.store.clear()

    # Index whatever is missing.
    indexed = set(engine.store.file_names())
    ingest_seconds = {}
    for key, rel in spec["documents"].items():
        path = PROJECT_DIR / rel
        if path.name in indexed:
            continue
        t0 = time.time()
        print(f"Indexing {path.name} ...", flush=True)
        n = engine.index_pdf(path.name, path.read_bytes())
        ingest_seconds[path.name] = round(time.time() - t0, 1)
        print(f"  {n} chunks in {ingest_seconds[path.name]}s", flush=True)

    rows = []
    for q in questions:
        engine.memory.clear()
        t0 = time.time()
        try:
            result = engine.ask(q["question"], debug=True) if _supports_debug(engine) else engine.ask(q["question"])
        except Exception as exc:  # keep going; record the failure
            result = {"answer": f"ERROR: {exc}", "sources": [], "grounded": False}
        latency = round(time.time() - t0, 2)
        g = grade(q, result)
        rows.append({"q": q, "result": result, "grade": g, "latency": latency})
        mark = "PASS" if g["passed"] else "FAIL"
        print(f"[{mark}] {q['id']:4} hit={g['hit']} mrr={_fmt(g['mrr'])} {latency}s  {q['question'][:70]}", flush=True)
        if args.verbose or not g["passed"]:
            print("       answer:", result.get("answer", "").replace("\n", " ")[:400])
            if g["missing"]:
                print("       missing:", g["missing"])
            if g["forbidden"]:
                print("       forbidden:", g["forbidden"])
            print("       sources:", [(s.get("page"), s.get("kind"), s.get("score")) for s in result.get("sources", [])][:8])

    report = summarize(rows, args, config, ingest_seconds)
    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"{stamp}-{args.label}.md").write_text(report, encoding="utf-8")
    (out_dir / f"{stamp}-{args.label}.json").write_text(
        json.dumps(
            [
                {"id": r["q"]["id"], "type": r["q"]["type"], "latency": r["latency"], **r["grade"],
                 "answer": r["result"].get("answer"), "sources": r["result"].get("sources")}
                for r in rows
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + report)


def _supports_debug(engine) -> bool:
    import inspect

    return "debug" in inspect.signature(engine.ask).parameters


def _fmt(x):
    return "-" if x is None else f"{x:.2f}"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(rows, args, config, ingest_seconds) -> str:
    lines = [f"# Eval: {args.label}", ""]
    lines.append(f"- Date: {datetime.now():%Y-%m-%d %H:%M}")
    lines.append(f"- Overrides: {', '.join(args.set) or '(none)'}")
    embedder = (
        f"local {config.LOCAL_EMBEDDING_MODEL}" if config.EMBEDDING_PROVIDER == "local"
        else f"gemini {config.EMBEDDING_MODEL} ({config.EMBEDDING_DIM}d)"
    )
    lines.append(f"- Chat model: {config.CHAT_MODEL} · Embedding: {embedder}")
    lines.append(f"- Pipeline: {args.pipeline}")
    if ingest_seconds:
        lines.append(f"- Ingest seconds: {ingest_seconds}")
    lines.append("")

    def block(name, subset):
        if not subset:
            return
        passed = sum(r["grade"]["passed"] for r in subset)
        strict = sum(r["grade"]["strict"] for r in subset)
        lines.append(
            f"| {name} | {passed}/{len(subset)} | {strict}/{len(subset)} "
            f"| {_fmt(_mean([r['grade']['hit'] for r in subset]))} "
            f"| {_fmt(_mean([r['grade']['mrr'] for r in subset]))} "
            f"| {_fmt(_mean([r['grade']['section_precision'] for r in subset]))} "
            f"| {sum(len(r['grade']['unsupported']) for r in subset)} "
            f"| {_fmt(_mean([r['latency'] for r in subset]))} |"
        )

    lines.append("| Slice | Pass | Strict pass | Hit@k | MRR | Section precision | Unsupported numbers | Avg latency s |")
    lines.append("|---|---|---|---|---|---|---|---|")
    block("**All**", rows)
    block("Sample (9)", [r for r in rows if r["q"]["type"] == "sample"])
    block("Held-out positives", [r for r in rows if r["q"]["type"] not in ("sample", "negative", "trap")])
    block("Traps", [r for r in rows if r["q"]["type"] == "trap"])
    block("Negatives", [r for r in rows if r["q"]["type"] == "negative"])
    for doc in ("gpo", "gep", "imac"):
        block(doc.upper(), [r for r in rows if r["q"]["doc"] == doc])
    lines.append("")
    lines.append("| Id | Type | Pass | Hit | MRR | Notes |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        g = r["grade"]
        notes = []
        if g["missing"]:
            notes.append("missing " + ", ".join(f"`{m}`" for m in g["missing"]))
        if g["forbidden"]:
            notes.append("forbidden " + ", ".join(f"`{m}`" for m in g["forbidden"]))
        if g["unsupported"]:
            notes.append("unsupported " + ", ".join(g["unsupported"]))
        lines.append(
            f"| {r['q']['id']} | {r['q']['type']} | {'✅' if g['passed'] else '❌'} | "
            f"{'-' if g['hit'] is None else ('✓' if g['hit'] else '✗')} | {_fmt(g['mrr'])} | "
            f"{'; '.join(notes).replace('|', '¦')} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
