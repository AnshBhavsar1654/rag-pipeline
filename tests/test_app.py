"""Web API tests: status, upload and the SSE chat stream (offline)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app import create_app
from tests.test_pipeline import DOC_TEXT, config, embeddings, make_pipeline  # noqa: F401


def events_from(response) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in response.text.split("\n\n")
        if line.startswith("data: ")
    ]


def test_upload_then_chat_streams_steps_tokens_and_sources(config, embeddings, tmp_path):
    config.ui.upload_dir = str(tmp_path / "uploads")
    pipeline = make_pipeline(config, embeddings, llm_responses=["It weighs 340 grams [1]."])
    client = TestClient(create_app(pipeline))

    assert client.get("/api/status").json()["ready"] is False
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 409

    res = client.post("/api/upload", files={"files": ("widgets.md", DOC_TEXT, "text/markdown")})
    assert res.status_code == 200 and res.json()["status"]["ready"] is True

    res = client.post("/api/chat", json={"message": "How heavy is the Zorblax-9?", "history": []})
    events = events_from(res)
    kinds = [e["kind"] for e in events]
    assert "step" in kinds and "token" in kinds and kinds[-1] == "done"
    assert {e["stage"] for e in events if e["kind"] == "step"} >= {"query_translation", "routing", "retrieval"}
    done = events[-1]
    assert done["answer"] == "It weighs 340 grams [1]."
    assert done["sources"][0]["label"].startswith("widgets.md")


def test_upload_rejects_unsupported_files(config, embeddings, tmp_path):
    config.ui.upload_dir = str(tmp_path / "uploads")
    client = TestClient(create_app(make_pipeline(config, embeddings)))
    res = client.post("/api/upload", files={"files": ("evil.exe", b"MZ", "application/octet-stream")})
    assert res.status_code == 400


def test_index_page_is_served(config, embeddings):
    client = TestClient(create_app(make_pipeline(config, embeddings)))
    assert "RAG Chatbot" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
