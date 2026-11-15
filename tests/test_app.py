"""Web API tests: background uploads, documents, the SSE chat stream (offline)."""

from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient
from langchain_core.language_models import FakeListChatModel

from app import create_app
from tests.test_pipeline import DOC_TEXT, config, embeddings, make_pipeline  # noqa: F401


def events_from(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: ")]


def upload_and_wait(client, name="widgets.md", text=DOC_TEXT) -> dict:
    res = client.post("/api/upload", files={"files": (name, text, "text/markdown")})
    assert res.status_code == 200
    job_id = res.json()["job"]["id"]
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("indexing job did not finish")


def make_client(config, embeddings, tmp_path, responses=("It weighs 340 grams [1].",)):
    config.ui.upload_dir = str(tmp_path / "uploads")
    pipeline = make_pipeline(config, embeddings, llm_responses=list(responses))
    return TestClient(create_app(pipeline)), pipeline


def test_upload_job_then_chat_streams_steps_tokens_and_sources(config, embeddings, tmp_path):
    client, _ = make_client(config, embeddings, tmp_path)
    assert client.get("/api/status").json()["ready"] is False
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 409

    job = upload_and_wait(client)
    assert job["state"] == "done" and job["message"].startswith("Indexed 1 file(s)")
    assert [d["name"] for d in client.get("/api/documents").json()] == ["widgets.md"]

    res = client.post("/api/chat", json={"message": "How heavy is the Zorblax-9?", "history": []})
    events = events_from(res)
    kinds = [e["kind"] for e in events]
    assert "step" in kinds and "token" in kinds and kinds[-1] == "done"
    assert all("ms" in e for e in events if e["kind"] == "step")
    done = events[-1]
    assert done["answer"] == "It weighs 340 grams [1]."
    assert done["sources"][0]["label"].startswith("widgets.md")
    assert done["ms"] >= 0 and done["model"]


def test_delete_document_removes_it_everywhere(config, embeddings, tmp_path):
    client, pipeline = make_client(config, embeddings, tmp_path)
    upload_and_wait(client)
    source = client.get("/api/documents").json()[0]["source"]
    assert client.delete("/api/documents", params={"source": source}).status_code == 200
    assert client.get("/api/documents").json() == []
    assert not (tmp_path / "uploads" / "widgets.md").exists()


def test_chat_errors_are_friendly_not_raw(config, embeddings, tmp_path):
    class Limited(FakeListChatModel):
        def _call(self, *a, **k):
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded for metric generate_content")

        def _stream(self, *a, **k):
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded for metric generate_content")

    client, pipeline = make_client(config, embeddings, tmp_path)
    upload_and_wait(client)
    pipeline._llm = Limited(responses=["x"])
    error = events_from(client.post("/api/chat", json={"message": "weight?"}))[-1]
    assert error["kind"] == "error"
    assert "429" not in error["message"] and "RESOURCE" not in error["message"]


def test_upload_rejects_unsupported_files(config, embeddings, tmp_path):
    client, _ = make_client(config, embeddings, tmp_path)
    res = client.post("/api/upload", files={"files": ("evil.exe", b"MZ", "application/octet-stream")})
    assert res.status_code == 400


def test_index_page_is_served(config, embeddings, tmp_path):
    client, _ = make_client(config, embeddings, tmp_path)
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
