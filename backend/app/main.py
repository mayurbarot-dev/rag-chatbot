"""FastAPI application: routes, validation, and lazy RAG initialization."""

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config
from .rag import RAG

log = logging.getLogger("uvicorn.error")

app = FastAPI(title="Technical Docs AI Assistant", version="1.0.0")

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


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


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
        "embedding_model": config.EMBEDDING_MODEL,
    }


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    engine = get_rag()
    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    results = []

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

        try:
            chunks = engine.index_pdf(name, data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("Indexing failed for %s", name)
            raise HTTPException(
                status_code=400, detail=f"Could not index '{name}': {exc}"
            ) from exc

        results.append({"file_name": name, "chunks": chunks})

    return {"indexed": results, "stats": engine.store.stats()}


@app.post("/api/ask")
def ask(body: AskRequest):
    engine = get_rag()
    try:
        return engine.ask(body.question)
    except Exception as exc:
        log.exception("Ask failed")
        raise HTTPException(status_code=500, detail=f"Failed to answer: {exc}") from exc


@app.get("/api/stats")
def stats():
    engine = get_rag()
    return {**engine.store.stats(), "files": engine.store.file_names()}


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
