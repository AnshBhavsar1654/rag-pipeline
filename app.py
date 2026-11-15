"""
RAG Pipeline — Web UI (FastAPI + plain HTML/JS)

Serves the chat page, streams every pipeline step and the answer as
Server-Sent Events, and indexes uploads as background jobs: a file becomes
searchable as soon as it is embedded, while graph extraction and RAPTOR
summaries continue in the background with progress the page can poll.

Usage:
    python app.py
    python app.py --config path/to/config.yaml --port 8000
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.chat.chatbot import format_history, messages_from_dicts
from src.generation.failover import friendly_error
from src.generation.generator import source_label
from src.indexing.loaders.loader import SUPPORTED_EXTENSIONS
from src.monitoring.telemetry import code_version, summarize
from src.pipeline.rag_pipeline import RAGPipeline
from src.stores.vectorstore import VectorStoreNotReadyError


WEB_DIR = Path(__file__).parent / "web"

STAGE_LABELS = {
    "queued": "Waiting to start",
    "loading": "Reading pages",
    "tables": "Extracting tables",
    "chunking": "Splitting into passages",
    "embedding": "Embedding passages",
    "graph": "Building the knowledge graph",
    "summaries": "Writing RAPTOR summaries",
    "done": "Done",
}


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@dataclass
class Job:
    """One upload being indexed."""
    id: str
    files: list[str]
    state: str = "queued"            # queued | indexing | enriching | done | error
    stage: str = "queued"
    label: str = STAGE_LABELS["queued"]
    done: int = 0
    total: int = 0
    message: str = ""
    sources: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    finished: float | None = None

    def update(self, stage: str, done: int, total: int) -> None:
        self.stage, self.done, self.total = stage, done, total
        self.label = STAGE_LABELS.get(stage, stage)


def load_pipeline(config_path: str | None) -> RAGPipeline:
    """Load the index, building it from the configured sources if there is none."""
    pipeline = RAGPipeline.from_config(config_path)
    try:
        pipeline.load()
    except VectorStoreNotReadyError as e:
        print(f"[INFO] {e}")
        print("[INFO] Ingesting documents from config...")
        try:
            pipeline.ingest()
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] Ingestion failed: {e}")
        if not pipeline.is_ready:
            print("[INFO] Starting with an empty index. Upload documents in the UI.")
    if pipeline.is_ready:
        try:
            pipeline.warm_up()
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Warm-up failed: {e}")
    return pipeline


def _event_json(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


def create_app(pipeline: RAGPipeline) -> FastAPI:
    app = FastAPI(title=pipeline.config.ui.title)
    jobs: dict[str, Job] = {}
    jobs_lock = threading.Lock()
    # One indexing job at a time; chats run alongside it.
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indexer")

    def active_jobs() -> list[Job]:
        with jobs_lock:
            return [j for j in jobs.values() if j.state in ("queued", "indexing", "enriching")]

    def status() -> dict:
        cfg = pipeline.config
        vectors = pipeline._vectorstore._collection.count() if pipeline._vectorstore is not None else 0
        return {
            "title": cfg.ui.title,
            "description": cfg.ui.description,
            "ready": pipeline.is_ready,
            "strategy": pipeline.strategy_name,
            "llm": cfg.llm.model,
            "fallback": cfg.llm.fallback.model if cfg.llm.fallback else None,
            "stores": {
                "vectorstore": vectors,
                "relational_db": len(pipeline.relational.table_names()),
                "graph_db": pipeline.graph.graph.number_of_edges(),
            },
            "features": {
                "indexing": f"{cfg.chunking.method} chunks, {cfg.indexing.multi_representation.replace('_', ' ')}"
                            + (", RAPTOR" if cfg.indexing.raptor.enabled else ""),
                "routing": " and ".join(
                    name for name, on in (("logical", cfg.routing.logical), ("semantic", cfg.routing.semantic)) if on
                ) or "off",
                "ranking": cfg.reranking.provider.replace("_", "-") if cfg.reranking.enabled else "off",
                "crag": cfg.retrieval.crag.enabled,
                "self_rag": cfg.generation.self_rag.enabled,
            },
            "extensions": sorted(SUPPORTED_EXTENSIONS),
            "jobs": [asdict(j) for j in active_jobs()],
            "models": pipeline.llm.status() if hasattr(pipeline.llm, "status") else [],
        }

    def chat_error_message(error: Exception) -> str:
        """Friendly text; when every model is paused, say when one is back."""
        models = pipeline.llm.status() if hasattr(pipeline.llm, "status") else []
        if models and all(m["paused_for"] > 0 for m in models):
            soonest = min(m["paused_for"] for m in models)
            when = (f"{round(soonest / 3600)} hours" if soonest >= 5400
                    else f"{max(1, round(soonest / 60))} minutes" if soonest >= 60 else f"{soonest} seconds")
            return (f"Every configured AI model has hit its usage limit. The next one is available "
                    f"again in about {when}. Add another provider's API key or a paid plan to avoid the wait.")
        return friendly_error(error)

    def run_job(job: Job, paths: list[str]) -> None:
        try:
            job.state = "indexing"
            result = pipeline.ingest_files(paths, progress=job.update, background=True)
            job.message, job.sources = result.message, result.sources
            if not result.loaded:
                job.state = "error"
                return
            if result.enrich is not None and (
                pipeline.config.graph_db.enabled or pipeline.config.indexing.raptor.enabled
            ):
                job.state = "enriching"
                result.enrich(job.update)
            job.update("done", 1, 1)
            job.state = "done"
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] Indexing job {job.id} failed: {e}")
            job.state = "error"
            job.message = (
                "Indexing stopped: " + friendly_error(e)
                if job.stage in ("graph", "summaries")
                else "Could not index the file. " + friendly_error(e)
            )
        finally:
            job.finished = time.time()

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/status")
    def get_status():
        return status()

    @app.get("/api/metrics")
    def metrics():
        """Telemetry per code version, plus this version's recent questions."""
        recent = [
            {k: r[k] for k in ("ts", "total_ms", "first_token_ms", "llm_calls", "failovers", "error", "stage_ms")}
            for r in pipeline.telemetry.rows("queries", limit=20)
            if r["version"] == code_version()
        ]
        return {"version": code_version(), "versions": summarize(pipeline.telemetry), "recent": recent}

    @app.get("/api/documents")
    def documents():
        enriching = {s for j in active_jobs() for s in j.sources}
        return [
            {**d, "enriching": d["source"] in enriching}
            for d in pipeline.list_sources()
        ]

    @app.delete("/api/documents")
    def delete_document(source: str):
        known = {d["source"] for d in pipeline.list_sources()}
        if source not in known:
            raise HTTPException(404, "That document is not in the index.")
        if any(source in j.sources for j in active_jobs()):
            raise HTTPException(409, "That document is still being indexed.")
        pipeline.remove_source(source)
        upload_dir = Path(pipeline.config.ui.upload_dir).resolve()
        path = Path(source).resolve()
        if path.is_file() and upload_dir in path.parents:
            path.unlink()
        return {"status": status()}

    @app.post("/api/upload")
    def upload(files: list[UploadFile] = File(...)):
        upload_dir = Path(pipeline.config.ui.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        paths, names, skipped = [], [], []
        for f in files:
            name = re.sub(r"[^\w.\- ]", "_", Path(f.filename or "upload").name)
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                skipped.append(name)
                continue
            target = upload_dir / name
            target.write_bytes(f.file.read())
            paths.append(str(target))
            names.append(name)
        if not paths:
            raise HTTPException(400, "Those file types aren't supported. Use: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        job = Job(id=uuid.uuid4().hex[:12], files=names)
        with jobs_lock:
            jobs[job.id] = job
        worker.submit(run_job, job, paths)
        return {"job": asdict(job), "skipped": skipped}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job.")
        return asdict(job)

    @app.post("/api/chat")
    def chat(request: ChatRequest):
        if not request.message.strip():
            raise HTTPException(400, "Empty message.")
        if not pipeline.is_ready:
            raise HTTPException(409, "No documents indexed yet. Add a document to start.")

        history = format_history(
            messages_from_dicts(request.history), pipeline.config.chat.memory_window,
        )

        def stream():
            started = time.perf_counter()
            last = started
            try:
                for event in pipeline.run(request.message, history):
                    now = time.perf_counter()
                    if event.kind == "step":
                        yield _event_json({
                            "kind": "step", "stage": event.stage, "message": event.message,
                            "ms": event.data.get("ms", int((now - last) * 1000)),
                            "at": int((now - started) * 1000),
                            "data": {k: v for k, v in event.data.items() if k != "ms"},
                        })
                        last = now
                    elif event.kind == "token":
                        yield _event_json({"kind": "token", "text": event.text})
                    elif event.kind == "reset":
                        yield _event_json({"kind": "reset"})
                    elif event.kind == "done":
                        yield _event_json({
                            "kind": "done",
                            "answer": event.answer,
                            **event.data,
                            "sources": [
                                {
                                    "n": i,
                                    "label": source_label(d),
                                    "type": d.metadata.get("type", "text"),
                                    "url": d.metadata.get("source")
                                    if str(d.metadata.get("source", "")).startswith("http") else None,
                                    "snippet": " ".join(d.page_content.split())[:400],
                                }
                                for i, d in enumerate(event.documents, start=1)
                            ],
                        })
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] Chat failed: {e}")
                yield _event_json({"kind": "error", "message": chat_error_message(e)})

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def main():
    parser = argparse.ArgumentParser(description="Launch the RAG Chatbot web UI.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file (default: config/config.yaml)")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    import uvicorn

    pipeline = load_pipeline(args.config)
    ui = pipeline.config.ui
    host, port = args.host or ui.host, args.port or ui.port
    print(f"\n[UI] Open http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}\n")
    uvicorn.run(create_app(pipeline), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
