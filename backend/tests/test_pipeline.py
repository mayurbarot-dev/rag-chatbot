"""End-to-end pipeline test with a deterministic stub in place of Gemini.

Proves parsing -> chunking -> embedding -> storage -> retrieval -> routing
without needing an API key. Embeddings are bag-of-words vectors, which is
crude but gives real, ordered cosine scores over the real sample PDFs.
"""
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(r"D:\rag-chatbot\backend")))

from app import config, rag  # noqa: E402

config.API_KEY = "stub-key"

VOCAB: dict[str, int] = {}
DIM = 4096


def fake_vec(text: str) -> list[float]:
    v = [0.0] * DIM
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        idx = VOCAB.setdefault(w, len(VOCAB) % DIM)
        v[idx] += 1.0
    return v


class StubRAG(rag.RAG):
    def __init__(self, tmp: Path):
        self.client = None
        self.store = rag.Store(path=tmp / "index.json")
        self.memory = rag.Memory(path=tmp / "memory.json")
        self.prompts: list[str] = []

    def _embed(self, texts, task_type):
        return [fake_vec(t) for t in texts]

    def _generate(self, prompt):
        self.prompts.append(prompt)
        if "TECHNICAL DOCUMENTATION CONTEXT" in prompt:
            return "GROUNDED_REPLY"
        return "GENERAL_REPLY"


tmp = Path(tempfile.mkdtemp())
r = StubRAG(tmp)
ok = fail = 0


def check(label, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label} {extra}")


print("\n== Indexing the real sample PDFs ==")
for name in ["orbital-api-reference.pdf", "platform-runbook.pdf"]:
    data = (Path(r"D:\rag-chatbot\sample_docs") / name).read_bytes()
    n = r.index_pdf(name, data)
    print(f"  {name}: {n} chunks")
check("index populated", len(r.store.items) == 6, r.store.stats())
check("stats correct", r.store.stats() == {"documents": 2, "chunks": 6}, r.store.stats())
check("page metadata kept", {i["metadata"]["page"] for i in r.store.items} == {1, 2})

print("\n== Empty / non-text PDF rejection ==")
try:
    r.index_pdf("blank.pdf", b"%PDF-1.4\nnot a real pdf")
    check("garbage pdf raises", False)
except Exception as e:
    check("garbage pdf raises", True, type(e).__name__)

print("\n== Route 1: memory question with nothing stored ==")
res = r.ask("What is my name?")
check("unknown name handled", "don't know your name" in res["answer"], res["answer"])
check("not grounded", res["grounded"] is False)

print("\n== Route 2: name setting ==")
res = r.ask("My name is Dev")
check("name acknowledged", "Dev" in res["answer"], res["answer"])
check("name persisted", r.memory.get_fact("name") == "Dev")
check("no LLM call needed", r.prompts == [])

print("\n== Route 1 again: memory question with name stored ==")
res = r.ask("what is my name?")
check("name recalled", res["answer"] == "Your name is Dev.", res["answer"])
check("no sources", res["sources"] == [])

print("\n== Route 3: casual greeting skips retrieval ==")
before = len(r.prompts)
res = r.ask("Hi")
check("general reply", res["answer"] == "GENERAL_REPLY", res["answer"])
check("general prompt used", "TECHNICAL DOCUMENTATION CONTEXT" not in r.prompts[-1])
check("name in general prompt", "Dev" in r.prompts[-1])
check("uploaded files listed", "orbital-api-reference.pdf" in r.prompts[-1])

print("\n== Route 9: grounded documentation question ==")
config.RELEVANCE_THRESHOLD = 0.10  # bag-of-words scores are lower than real embeddings
res = r.ask("What is the default rate limit per API key?")
check("grounded", res["grounded"] is True, res)
check("answer from grounded prompt", res["answer"] == "GROUNDED_REPLY")
check("sources returned", len(res["sources"]) > 0)
top = res["sources"][0]
check("top source is the API reference", top["file_name"] == "orbital-api-reference.pdf", top)
check("top source is page 1", top["page"] == 1, top)
check("score rounded to 3dp", round(top["score"], 3) == top["score"], top)
check("scores descending", all(
    res["sources"][i]["score"] >= res["sources"][i + 1]["score"]
    for i in range(len(res["sources"]) - 1)), res["sources"])
p = r.prompts[-1]
check("prompt has SOURCE markers", "[SOURCE 1]" in p and "File:" in p and "Page:" in p)
check("name is not evidence", "NOT documentation evidence" in p)

print("\n== Route 9: second grounded question hits the runbook ==")
res = r.ask("Who owns service ID 2 in the registry?")
check("grounded", res["grounded"] is True, res)
check("runbook is top source", res["sources"][0]["file_name"] == "platform-runbook.pdf", res["sources"][0])
check("runbook page 2", res["sources"][0]["page"] == 2, res["sources"][0])

print("\n== Route 8: exact-ID guardrail ==")
res = r.ask("Who owns service ID 006 in the registry?")
check("refuses missing ID", res["grounded"] is False, res)
check("explains rather than guessing", "identifier" in res["answer"].lower(), res["answer"])
check("no sources cited", res["sources"] == [])

print("\n== Route 7: below relevance threshold ==")
config.RELEVANCE_THRESHOLD = 0.99
res = r.ask("What is the document page section about quantum gardening techniques?")
check("falls back", res["grounded"] is False, res)
check("says nothing relevant", "couldn't find" in res["answer"], res["answer"])
config.RELEVANCE_THRESHOLD = 0.10

print("\n== Route 4: empty index ==")
r2 = StubRAG(Path(tempfile.mkdtemp()))
res = r2.ask("What is the rate limit in the document?")
check("general answer when nothing indexed", res["answer"] == "GENERAL_REPLY", res)
check("prompt says none uploaded", "No documents have been uploaded" in r2.prompts[-1])

print("\n== Memory: history trimming and separation ==")
check("history capped at 12", len(r.memory.data["history"]) == 12, len(r.memory.data["history"]))
check("recent() returns 10", len(r.memory.recent()) == 10)
r.store.clear()
check("clearing docs keeps memory", r.memory.get_fact("name") == "Dev")
check("docs actually cleared", r.store.stats() == {"documents": 0, "chunks": 0})
r.memory.clear()
check("clearing memory resets facts", r.memory.get_fact("name") is None)

print(f"\n{'='*46}\n  {ok} passed, {fail} failed\n{'='*46}")
sys.exit(1 if fail else 0)
