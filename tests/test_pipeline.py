"""Offline tests: fake embeddings and a scripted LLM, no network or API keys."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from src.chat.chatbot import format_history, messages_from_gradio
from src.config import RAGConfig
from src.generation.generator import SOURCES_HEADER, generate_answer
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.utils import parse_query_list, reciprocal_rank_fusion
from src.vectorstore.store import (
    VectorStoreNotReadyError,
    build_vectorstore,
    load_vectorstore,
)


DOC_TEXT = """\
# Widgets

The Zorblax-9 widget runs on a 12 volt supply and weighs 340 grams.

## Maintenance

Clean the Zorblax-9 intake filter every 30 days with isopropyl alcohol.
"""


@pytest.fixture
def embeddings():
    return DeterministicFakeEmbedding(size=32)


@pytest.fixture
def config(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "widgets.md").write_text(DOC_TEXT, encoding="utf-8")
    return RAGConfig(
        documents={"sources": [str(docs)]},
        chunking={"chunk_size": 40, "chunk_overlap": 0},
        vectorstore={"persist_directory": str(tmp_path / "db")},
        retrieval={"strategy": "simple", "top_k": 3},
        reranking={"enabled": False},
        extraction={"enabled": False},
    )


def make_pipeline(config, embeddings, llm_responses=("An answer [1].",)):
    pipeline = RAGPipeline(config)
    pipeline._embeddings = embeddings
    pipeline._llm = FakeListChatModel(responses=list(llm_responses))
    return pipeline


# --- helpers ----------------------------------------------------------------

def test_parse_query_list_strips_numbering_bullets_and_preamble():
    text = 'Here are 3 queries:\n1. first one\n- "second one"\n\n3) third one\n2. first one'
    assert parse_query_list(text) == ["first one", "second one", "third one"]
    assert parse_query_list(text, limit=2) == ["first one", "second one"]


def test_rrf_rewards_documents_ranked_well_in_several_lists():
    a, b, c = (Document(page_content=x, id=x) for x in "abc")
    fused = reciprocal_rank_fusion([[a, b, c], [b, c, a], [b, a]])
    assert [d.page_content for d in fused] == ["b", "a", "c"]


def test_bm25_finds_exact_terms_dense_search_misses(tmp_path, embeddings):
    docs = [
        Document(page_content="General notes about the product line."),
        Document(page_content="Part number QX-77812 ships in blue."),
        Document(page_content="Warranty terms and conditions."),
    ]
    vs = build_vectorstore(
        RAGConfig(vectorstore={"persist_directory": str(tmp_path)}).vectorstore,
        embeddings, "fake", docs,
    )
    from src.vectorstore.store import all_documents

    retriever = HybridRetriever(vs, all_documents(vs), k=2, fetch_k=3)
    assert "QX-77812" in retriever.invoke("what colour is QX-77812")[0].page_content


# --- vector store -----------------------------------------------------------

def test_reingest_replaces_instead_of_duplicating(config, embeddings):
    pipeline = make_pipeline(config, embeddings)
    pipeline.ingest()
    first = pipeline._vectorstore._collection.count()
    pipeline.ingest()
    assert first > 0
    assert pipeline._vectorstore._collection.count() == first


def test_reingest_keeps_files_uploaded_through_the_ui(config, embeddings, tmp_path):
    upload = tmp_path / "upload.md"
    upload.write_text("Uploaded note: the Q3 offsite is in Lisbon.", encoding="utf-8")

    pipeline = make_pipeline(config, embeddings)
    pipeline.ingest()
    pipeline.ingest_files([str(upload)])
    with_upload = pipeline._vectorstore._collection.count()

    rerun = make_pipeline(config, embeddings)
    rerun.ingest()
    assert rerun._vectorstore._collection.count() == with_upload

    rerun.ingest(rebuild=True)
    assert rerun._vectorstore._collection.count() < with_upload


def test_model_change_reembeds_stored_chunks_instead_of_dropping_them(
    config, embeddings, tmp_path
):
    upload = tmp_path / "only_copy.md"
    upload.write_text("The vault code is 4417.", encoding="utf-8")
    make_pipeline(config, embeddings).ingest_files([str(upload)])
    upload.unlink()  # the index now holds the only copy

    config.embedding.model = "some-newer-model"
    pipeline = make_pipeline(config, DeterministicFakeEmbedding(size=16))
    with pytest.raises(VectorStoreNotReadyError):
        pipeline.load()

    pipeline.ingest()  # what app.py does when load() fails
    texts = [d.page_content for d in pipeline._base_retriever._documents]
    assert any("vault code is 4417" in t for t in texts)
    assert list(tmp_path.glob("db.backup-*")), "old index should be backed up"
    load_vectorstore(config.vectorstore, pipeline.embeddings, config.embedding.signature)


def test_load_rejects_index_built_with_other_embedding_model(config, embeddings):
    make_pipeline(config, embeddings).ingest()

    load_vectorstore(config.vectorstore, embeddings, config.embedding.signature)
    with pytest.raises(VectorStoreNotReadyError, match="different|was built"):
        load_vectorstore(config.vectorstore, embeddings, "huggingface:some-other-model")


def test_load_missing_store_raises_not_ready(config, embeddings):
    with pytest.raises(VectorStoreNotReadyError):
        make_pipeline(config, embeddings).load()


# --- pipeline ---------------------------------------------------------------

def test_upload_into_empty_store_works_and_is_idempotent(config, embeddings, tmp_path):
    upload = tmp_path / "upload.md"
    upload.write_text(DOC_TEXT, encoding="utf-8")

    pipeline = make_pipeline(config, embeddings)
    msg = pipeline.ingest_files([str(upload)])
    assert msg.startswith("Successfully ingested 1 file(s)")
    assert pipeline.is_ready

    count = pipeline._vectorstore._collection.count()
    pipeline.ingest_files([str(upload)])
    assert pipeline._vectorstore._collection.count() == count


def test_query_caps_context_and_numbers_sources(config, embeddings):
    pipeline = make_pipeline(config, embeddings)
    pipeline.ingest()

    answer, docs = pipeline.query("How heavy is the Zorblax-9?")
    assert answer == "An answer [1]."
    assert 0 < len(docs) <= config.retrieval.top_k

    response = pipeline.format_response(answer, docs)
    assert SOURCES_HEADER in response
    assert "**[1]** widgets.md" in response


@pytest.mark.parametrize("strategy", ["multi_query", "rag_fusion", "hyde", "step_back", "decomposition"])
def test_llm_strategies_retrieve_and_cap(config, embeddings, strategy):
    config.retrieval.strategy = strategy
    pipeline = make_pipeline(
        config, embeddings,
        llm_responses=["1. Zorblax-9 weight\n2. Zorblax-9 maintenance"] * 5,
    )
    pipeline.ingest()
    docs = pipeline.retrieve("Tell me about the Zorblax-9")
    assert 0 < len(docs) <= config.retrieval.top_k


def test_strategy_falls_back_to_plain_search_when_llm_fails(config, embeddings):
    class BrokenLLM(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise RuntimeError("503 UNAVAILABLE")

    config.retrieval.strategy = "rag_fusion"
    pipeline = make_pipeline(config, embeddings)
    pipeline._llm = BrokenLLM(responses=["unused"])
    pipeline.ingest()
    assert pipeline.retrieve("Zorblax-9 filter")


def test_follow_up_is_rewritten_before_search(config, embeddings, monkeypatch):
    pipeline = make_pipeline(
        config, embeddings,
        llm_responses=["How often should the Zorblax-9 filter be cleaned?", "Every 30 days [1]."],
    )
    pipeline.ingest()

    searched: list[str] = []
    original = pipeline._retriever.retrieve
    monkeypatch.setattr(
        pipeline._retriever, "retrieve", lambda q: searched.append(q) or original(q)
    )

    answer, _ = pipeline.query("How often?", chat_history="Human: Tell me about the filter")
    assert searched == ["How often should the Zorblax-9 filter be cleaned?"]
    assert answer == "Every 30 days [1]."


def test_system_prompt_with_braces_does_not_break_template():
    llm = FakeListChatModel(responses=["ok"])
    answer = generate_answer(
        llm, "q", [Document(page_content="ctx {not a variable}")],
        system_prompt="Reply as JSON like {\"answer\": ...}",
    )
    assert answer == "ok"


# --- chat -------------------------------------------------------------------

def test_gradio_history_drops_source_lists_and_respects_window():
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": f"a1\n\n{SOURCES_HEADER}\n- **[1]** x.md"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": [{"type": "text", "text": "a2"}]},
    ]
    messages = messages_from_gradio(history)
    assert [m.content for m in messages] == ["q1", "a1", "q2", "a2"]
    assert format_history(messages, memory_window=1) == "Human: q2\nAssistant: a2"
