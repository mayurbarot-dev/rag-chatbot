"""End-to-end pipeline test with a deterministic stand-in for Gemini.

Proves parsing -> chunking -> embedding -> storage -> retrieval -> rerank ->
routing without an API key. Embeddings are bag-of-words vectors and the
"reranker" scores passages by word overlap: crude, but it gives real, ordered
scores over the real sample PDFs.
"""
import os
import re
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

from app import config, rag  # noqa: E402
from app.store import Store, tokenize  # noqa: E402

DIM = 4096


def fake_vec(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for w in tokenize(text):
        v[hash(w) % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeLLM:
    embedding_namespace = "fake|bow"

    def __init__(self):
        self.prompts: list[str] = []

    def embed_documents(self, texts, progress=None):
        if progress:
            progress(len(texts), len(texts))
        return np.vstack([fake_vec(t) for t in texts])

    def embed_query(self, text):
        return fake_vec(text)

    def generate(self, prompt, temperature=0.2):
        self.prompts.append(prompt)
        return "GROUNDED_REPLY" if "SOURCES:" in prompt else "GENERAL_REPLY"

    def generate_json(self, prompt, schema, temperature=0.0):
        question = re.search(r"QUESTION: (.*)", prompt).group(1)
        q = set(tokenize(question))
        scores = []
        for m in re.finditer(r"^\[(\d+)\] (.*?)(?=^\[\d+\] |\Z)", prompt.split("PASSAGES:", 1)[1], re.S | re.M):
            overlap = len(q & set(tokenize(m.group(2))))
            scores.append({"id": int(m.group(1)), "score": 3 if overlap >= 3 else 2 if overlap == 2 else 0})
        return {"scores": scores}

    def describe_image(self, png, prompt, schema):
        return {"visuals": []}


def make_engine(tmp: Path):
    return rag.RAG(llm=FakeLLM(), store=Store(tmp / "index"), memory=rag.Memory(tmp / "memory.json"))


r = make_engine(TMP / "a")
ok = fail = 0


def check(label, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label} {extra}")


SAMPLES = PROJECT / "sample_docs"

print("\n== Indexing the real sample PDFs ==")
counts = {}
for name in ["orbital-api-reference.pdf", "platform-runbook.pdf"]:
    counts[name] = r.index_pdf(name, (SAMPLES / name).read_bytes())
    print(f"  {name}: {counts[name]} chunks")
stats = r.store.stats()
check("two documents indexed", stats["documents"] == 2, stats)
check("chunks stored with vectors", len(r.store.chunks) == len(r.store.matrix) > 0, stats)
check("page metadata kept", {c["page"] for c in r.store.chunks} == {1, 2})
check("every chunk has a header and section path", all(c["header"] and "section_path" in c for c in r.store.chunks))
check("search text embeds the header", all(c["search_text"].startswith(c["header"]) for c in r.store.chunks))

print("\n== Index survives a restart ==")
reloaded = Store(TMP / "a" / "index")
check("chunks reloaded", len(reloaded.chunks) == len(r.store.chunks))
check("vectors reloaded", reloaded.matrix.shape == r.store.matrix.shape)
check("bm25 rebuilt", reloaded.bm25.n == len(reloaded.chunks))

print("\n== Re-uploading a file replaces its chunks ==")
before = len(r.store.chunks)
r.index_pdf("platform-runbook.pdf", (SAMPLES / "platform-runbook.pdf").read_bytes())
check("no duplicates after re-upload", len(r.store.chunks) == before, r.store.stats())
check("every chunk still distinct", len({c["text"] for c in r.store.chunks}) == len(r.store.chunks))
check("other file untouched",
      sum(1 for c in r.store.chunks if c["file_name"] == "orbital-api-reference.pdf") == counts["orbital-api-reference.pdf"])

print("\n== Bad PDF rejection ==")
try:
    r.index_pdf("blank.pdf", b"%PDF-1.4\nnot a real pdf")
    check("garbage pdf raises", False)
except ValueError as e:
    check("garbage pdf raises ValueError", True, str(e))

print("\n== Route 1: memory question with nothing stored ==")
res = r.ask("What is my name?")
check("unknown name handled", "don't know your name" in res["answer"], res["answer"])
check("not grounded", res["grounded"] is False)

print("\n== Route 2: name setting ==")
res = r.ask("My name is Dev")
check("name acknowledged", "Dev" in res["answer"], res["answer"])
check("name persisted", r.memory.get_fact("name") == "Dev")
check("no LLM call needed", r.llm.prompts == [])

print("\n== Route 1 again: memory question with name stored ==")
res = r.ask("what is my name?")
check("name recalled", res["answer"] == "Your name is Dev.", res["answer"])
check("no sources", res["sources"] == [])

print("\n== Name given and asked in one message ==")
res = r.ask("Hi, my name is Mayur. What is my name?")
check("name saved and recalled together", res["answer"] == "Your name is Mayur.", res["answer"])
r.memory.set_fact("name", "Dev")

print("\n== Route 3: casual greeting skips retrieval ==")
res = r.ask("Hi")
check("general reply", res["answer"] == "GENERAL_REPLY", res["answer"])
check("general prompt used", "SOURCES:" not in r.llm.prompts[-1])
check("name in general prompt", "Dev" in r.llm.prompts[-1])
check("uploaded files listed", "orbital-api-reference.pdf" in r.llm.prompts[-1])

print("\n== Route 9: grounded question ==")
res = r.ask("What is the default rate limit per API key?", debug=True)
check("grounded", res["grounded"] is True, res.get("answer"))
check("answer from grounded prompt", res["answer"] == "GROUNDED_REPLY")
check("sources returned", len(res["sources"]) > 0)
check("API reference page 1 is cited",
      any(s["file_name"] == "orbital-api-reference.pdf" and s["page"] == 1 for s in res["sources"]), res["sources"])
check("sources carry section and kind", all("section_path" in s and "kind" in s for s in res["sources"]))
check("grounding report attached", res.get("grounding", {}).get("checked") is True, res.get("grounding"))
check("debug shows the query plan", "plan" in res.get("debug", {}))
p = r.llm.prompts[-1]
check("prompt has SOURCE labels with pages", "[SOURCE 1]" in p and "(PDF p" in p)
check("name is not evidence", "NOT document evidence" in p)

print("\n== Route 9: second grounded question hits the runbook ==")
res = r.ask("Who owns service ID 2 in the registry?")
check("grounded", res["grounded"] is True, res.get("answer"))
check("runbook page 2 is cited",
      any(s["file_name"] == "platform-runbook.pdf" and s["page"] <= 2 <= s["page_end"] for s in res["sources"]),
      res["sources"])

print("\n== Route 8: exact-ID guardrail ==")
res = r.ask("Who owns service ID 006 in the registry?")
check("refuses missing ID", res["grounded"] is False, res)
check("explains rather than guessing", "identifier" in res["answer"].lower(), res["answer"])
check("no sources cited", res["sources"] == [])

print("\n== Route 7: reranker finds nothing relevant ==")
res = r.ask("What is the recommended soil acidity for quantum gardening?")
check("falls back", res["grounded"] is False, res)
check("says nothing relevant", "couldn't find" in res["answer"], res["answer"])

print("\n== Route 7 with RERANK=off: cosine gate ==")
config.RERANK = "off"
config.RELEVANCE_THRESHOLD = 0.99
res = r.ask("What is the recommended soil acidity for quantum gardening?")
check("cosine gate refuses", res["grounded"] is False, res)
config.RELEVANCE_THRESHOLD = 0.05
res = r.ask("What is the default rate limit per API key?")
check("cosine gate admits a real question", res["grounded"] is True, res.get("answer"))
config.RERANK = "llm"

print("\n== Route 4: empty index ==")
r2 = make_engine(TMP / "b")
res = r2.ask("What is the rate limit in the document?")
check("general answer when nothing indexed", res["answer"] == "GENERAL_REPLY", res)
check("prompt says none uploaded", "No documents have been uploaded" in r2.llm.prompts[-1])

print("\n== Grounding retry ==")


class NumberLLM(FakeLLM):
    def generate(self, prompt, temperature=0.2):
        self.prompts.append(prompt)
        if "SOURCES:" not in prompt:
            return "GENERAL_REPLY"
        return "The limit is 120 requests." if "IMPORTANT: a previous draft" in prompt else "The limit is 987654 requests."


r.llm = NumberLLM()
res = r.ask("What is the default rate limit per API key?")
check("retried after an unsupported figure", res["grounding"]["retried"] is True, res["grounding"])
check("unsupported figure removed by retry", "987654" not in res["answer"], res["answer"])
r.llm = FakeLLM()

print("\n== Memory: history trimming and separation ==")
check("history capped at 12", len(r.memory.data["history"]) == 12, len(r.memory.data["history"]))
check("recent() returns 10", len(r.memory.recent()) == 10)
r.store.clear()
check("clearing docs keeps memory", r.memory.get_fact("name") == "Dev")
check("docs actually cleared", r.store.stats()["chunks"] == 0, r.store.stats())
r.memory.clear()
check("clearing memory resets facts", r.memory.get_fact("name") is None)

print(f"\n{'=' * 46}\n  {ok} passed, {fail} failed\n{'=' * 46}")
sys.exit(1 if fail else 0)
