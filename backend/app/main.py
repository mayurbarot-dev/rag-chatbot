"""FastAPI application: routes, validation, background ingestion, lazy RAG init."""

import logging

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config
from .jobs import JobManager
from .rag import RAG

log = logging.getLogger("uvicorn.error")

app = FastAPI(title="Document Q&A Assistant", version="2.0.0")

# Built on first use, not at import time, so /api/health can report key status
# without ever touching Gemini.
rag: RAG | None = None


def get_rag() -> RAG:
    global rag
    if not config.has_api_key():
        raise HTTPException(
            status_code=503,
            detail=(
                "No Gemini API key configured. Add GOOGLE_API_KEY to backend/.env "
                "and restart. Get a free key at https://aistudio.google.com/app/apikey"
            ),
        )
    if rag is None:
        rag = RAG()
    return rag


jobs = JobManager(get_rag)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    doc_ids: list[str] | None = None
    debug: bool = False


@app.get("/")
def index():
    if not config.FRONTEND_FILE.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    # Without these, a browser keeps serving a stale index.html during UI work.
    return FileResponse(
        config.FRONTEND_FILE,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "api_key_configured": config.has_api_key(),
        "chat_model": config.CHAT_MODEL,
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "embedding_model": (
            config.LOCAL_EMBEDDING_MODEL if config.EMBEDDING_PROVIDER == "local" else config.EMBEDDING_MODEL
        ),
        "vision_mode": config.VISION_MODE,
        "rerank": config.RERANK,
    }


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), wait: bool = Query(False)):
    """Queue PDFs for indexing. Returns job ids; poll /api/jobs/{id}.

    `?wait=true` blocks until every job finishes (scripts and tests).
    """
    get_rag()  # fail fast with 503 when no key is configured
    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    accepted = []
    payloads = []

    for upload_file in files:
        name = (upload_file.filename or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="A file was uploaded without a name.")
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"'{name}' is not a PDF.")
        data = await upload_file.read()
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"'{name}' is larger than the {config.MAX_FILE_SIZE_MB} MB limit.",
            )
        if not data:
            raise HTTPException(status_code=400, detail=f"'{name}' is empty.")
        if not data.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail=f"'{name}' does not look like a PDF file.")
        payloads.append((name, data))

    for name, data in payloads:
        accepted.append(jobs.submit(name, data))

    if wait:
        finished = [jobs.wait(j["id"]) for j in accepted]
        failed = [j for j in finished if j and j["status"] == "error"]
        if failed:
            raise HTTPException(status_code=400, detail=f"Could not index '{failed[0]['file_name']}': {failed[0]['error']}")
        engine = get_rag()
        return {
            "indexed": [{"file_name": j["file_name"], "chunks": j["chunks"]} for j in finished],
            "stats": engine.store.stats(),
        }
    return {"jobs": accepted}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    return job


@app.post("/api/ask")
def ask(body: AskRequest):
    engine = get_rag()
    try:
        return engine.ask(body.question, doc_ids=body.doc_ids or None, debug=body.debug)
    except Exception as exc:
        log.exception("Ask failed")
        raise HTTPException(status_code=500, detail=f"Failed to answer: {exc}") from exc


@app.get("/api/stats")
def stats():
    engine = get_rag()
    out = {**engine.store.stats(), "files": engine.store.file_names()}
    if engine.store.legacy_index:
        out["warning"] = (
            "An index from the previous version was found but is not compatible. "
            "Upload your documents again to re-index them."
        )
    return out


@app.get("/api/documents")
def documents():
    engine = get_rag()
    docs = engine.store.documents()
    for d in docs:
        d.pop("glossary", None)
        d.pop("aliases", None)
    return {"documents": docs}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    engine = get_rag()
    if doc_id not in engine.store.docs:
        raise HTTPException(status_code=404, detail="Unknown document.")
    removed = engine.store.remove_document(doc_id)
    return {"status": "deleted", "chunks_removed": removed, "stats": engine.store.stats()}


@app.delete("/api/knowledge-base")
def clear_knowledge_base():
    """Clears indexed documents only. User memory is untouched."""
    engine = get_rag()
    engine.store.clear()
    return {"status": "cleared", "stats": engine.store.stats()}


@app.delete("/api/memory")
def clear_memory():
    """Clears remembered facts and history only. Documents are untouched."""
    engine = get_rag()
    engine.memory.clear()
    return {"status": "cleared"}
