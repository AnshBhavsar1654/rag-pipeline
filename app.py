"""
RAG Pipeline — Web UI (FastAPI + plain HTML/JS)

Serves a chat page that streams every pipeline step (query translation,
routing, query construction, retrieval, generation) and the answer as
Server-Sent Events, plus document upload.

Usage:
    python app.py
    python app.py --config path/to/config.yaml --port 8000
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.chat.chatbot import format_history, messages_from_dicts
from src.generation.generator import source_label
from src.indexing.loaders.loader import SUPPORTED_EXTENSIONS
from src.pipeline.rag_pipeline import RAGPipeline
from src.stores.vectorstore import VectorStoreNotReadyError


WEB_DIR = Path(__file__).parent / "web"


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


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
        except Exception as e:
            print(f"[ERROR] Ingestion failed: {e}")
        if not pipeline.is_ready:
            print("[INFO] Starting with an empty index. Upload documents in the UI.")
    return pipeline


def create_app(pipeline: RAGPipeline) -> FastAPI:
    app = FastAPI(title=pipeline.config.ui.title)
    # The pipeline (Chroma, the graph, model caches) is not thread-safe.
    lock = threading.Lock()

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
                "indexing": f"{cfg.chunking.method} chunks · {cfg.indexing.multi_representation}"
                            + (" · RAPTOR" if cfg.indexing.raptor.enabled else ""),
                "routing": " · ".join(
                    name for name, on in (("logical", cfg.routing.logical), ("semantic", cfg.routing.semantic)) if on
                ) or "off",
                "ranking": cfg.reranking.provider if cfg.reranking.enabled else "off",
                "crag": cfg.retrieval.crag.enabled,
                "self_rag": cfg.generation.self_rag.enabled,
            },
            "extensions": sorted(SUPPORTED_EXTENSIONS),
        }

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/status")
    def get_status():
        return status()

    @app.post("/api/upload")
    def upload(files: list[UploadFile] = File(...)):
        upload_dir = Path(pipeline.config.ui.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for f in files:
            name = re.sub(r"[^\w.\- ]", "_", Path(f.filename or "upload").name)
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            target = upload_dir / name
            target.write_bytes(f.file.read())
            paths.append(str(target))
        if not paths:
            raise HTTPException(400, "No supported files. Use: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        with lock:
            try:
                message = pipeline.ingest_files(paths)
            except Exception as e:
                raise HTTPException(500, f"Ingestion failed: {e}")
        return {"message": message, "status": status()}

    @app.post("/api/chat")
    def chat(request: ChatRequest):
        if not request.message.strip():
            raise HTTPException(400, "Empty message.")
        if not pipeline.is_ready:
            raise HTTPException(409, "No documents indexed yet. Upload documents first.")

        history = format_history(
            messages_from_dicts(request.history), pipeline.config.chat.memory_window,
        )

        def stream():
            with lock:
                try:
                    for event in pipeline.run(request.message, history):
                        payload = {"kind": event.kind}
                        if event.kind == "step":
                            payload.update(stage=event.stage, message=event.message, data=event.data)
                        elif event.kind == "token":
                            payload["text"] = event.text
                        elif event.kind == "done":
                            payload["answer"] = event.answer
                            payload["sources"] = [
                                {
                                    "n": i,
                                    "label": source_label(d),
                                    "type": d.metadata.get("type", "text"),
                                    "url": d.metadata.get("source") if str(d.metadata.get("source", "")).startswith("http") else None,
                                    "snippet": " ".join(d.page_content.split())[:400],
                                }
                                for i, d in enumerate(event.documents, start=1)
                            ]
                        yield f"data: {json.dumps(payload, default=str)}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'kind': 'error', 'message': str(e)})}\n\n"

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
