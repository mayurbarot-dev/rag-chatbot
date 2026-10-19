"""Background ingestion with progress.

Indexing the 200-page GEP report takes minutes (about 700 chunks to embed and
up to 30 chart pages to transcribe), far longer than a browser should hold an
upload request open. Uploads are queued; one worker indexes them in order and
publishes per-stage progress that the UI polls.
"""

import logging
import queue
import threading
import time
import uuid
from typing import Callable

log = logging.getLogger("uvicorn.error")

STAGE_LABELS = {
    "queued": "Waiting",
    "parsing": "Reading PDF",
    "layout": "Rebuilding tables and structure",
    "vision": "Transcribing charts",
    "chunking": "Chunking",
    "embedding": "Embedding",
    "saving": "Saving index",
    "done": "Indexed",
    "error": "Failed",
}


class JobManager:
    def __init__(self, get_engine: Callable):
        self.get_engine = get_engine
        self.jobs: dict[str, dict] = {}
        self.queue: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self._run, name="ingest-worker", daemon=True)
        self.worker.start()

    def submit(self, file_name: str, data: bytes) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id, "file_name": file_name, "status": "queued", "stage": "queued",
            "stage_label": STAGE_LABELS["queued"], "done": 0, "total": 0, "chunks": None,
            "error": None, "warnings": [], "created": time.time(), "finished": None,
        }
        with self.lock:
            self.jobs[job_id] = job
            # Keep the registry bounded; finished jobs older than the newest 50 go.
            if len(self.jobs) > 50:
                for old in sorted(self.jobs.values(), key=lambda j: j["created"])[:-50]:
                    if old["status"] in ("done", "error"):
                        self.jobs.pop(old["id"], None)
        self.queue.put((job_id, data))
        return dict(job)

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def wait(self, job_id: str, timeout: float = 1800) -> dict | None:
        end = time.time() + timeout
        while time.time() < end:
            job = self.get(job_id)
            if job is None or job["status"] in ("done", "error"):
                return job
            time.sleep(0.25)
        return self.get(job_id)

    def _update(self, job_id: str, **fields) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job:
                job.update(fields)
                if "stage" in fields:
                    job["stage_label"] = STAGE_LABELS.get(fields["stage"], fields["stage"])

    def _run(self) -> None:
        while True:
            job_id, data = self.queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            self._update(job_id, status="running", stage="parsing")

            def progress(stage: str, done: int = 0, total: int = 0, job_id=job_id):
                self._update(job_id, stage=stage, done=done, total=total)

            try:
                engine = self.get_engine()
                chunks = engine.index_pdf(job["file_name"], data, progress)
                warnings = []
                for info in engine.store.docs.values():
                    if info.get("file_name") == job["file_name"]:
                        warnings = info.get("warnings", [])
                self._update(job_id, status="done", stage="done", chunks=chunks,
                             warnings=warnings, finished=time.time())
            except Exception as exc:
                log.exception("Indexing failed for %s", job["file_name"])
                self._update(job_id, status="error", stage="error", error=str(exc), finished=time.time())
            finally:
                self.queue.task_done()
