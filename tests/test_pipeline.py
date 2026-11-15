"""Offline tests: fake embeddings and a scripted LLM, no network or API keys."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from src.chat.chatbot import format_history, messages_from_dicts
from src.config import RAGConfig
from src.generation.generator import SOURCES_HEADER, generate_answer
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.utils import parse_query_list, reciprocal_rank_fusion
from src.stores.vectorstore import (
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
        relational_db={"path": str(tmp_path / "rel.sqlite")},
        graph_db={"enabled": False, "path": str(tmp_path / "graph.json")},
        indexing={"docstore_path": str(tmp_path / "docstore.sqlite"), "raptor": {"enabled": False}},
        retrieval={"strategy": "simple", "top_k": 3, "crag": {"enabled": False}},
        reranking={"enabled": False},
        generation={"self_rag": {"enabled": False}},
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
    from src.stores.vectorstore import all_documents

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
    result = pipeline.ingest_files([str(upload)])
    assert result.message.startswith("Indexed 1 file(s)") and result.loaded == ["upload.md"]
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
    original = pipeline._search
    monkeypatch.setattr(
        pipeline, "_search", lambda queries, *a: searched.extend(queries) or original(queries, *a)
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

def test_ui_history_drops_source_lists_and_respects_window():
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": f"a1\n\n{SOURCES_HEADER}\n- **[1]** x.md"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": [{"type": "text", "text": "a2"}]},
    ]
    messages = messages_from_dicts(history)
    assert [m.content for m in messages] == ["q1", "a1", "q2", "a2"]
    assert format_history(messages, memory_window=1) == "Human: q2\nAssistant: a2"


# --- retrieval refinement and generation checks ----------------------------

def test_crag_drops_irrelevant_documents_and_re_retrieves(config, embeddings, monkeypatch):
    config.retrieval.crag.enabled = True
    config.indexing.multi_representation = "none"
    web_doc = Document(page_content="Filters: clean monthly.", metadata={"source": "https://x.test"})
    monkeypatch.setattr("src.pipeline.rag_pipeline.web_search", lambda q, n: [web_doc])
    # grade: nothing relevant → rewrite → grade the web result: keep #1 → answer
    pipeline = make_pipeline(
        config, embeddings,
        llm_responses=["none", "Zorblax-9 intake filter cleaning", "1", "Every 30 days [1]."],
    )
    pipeline.ingest()
    events = list(pipeline.run("filter?"))
    steps = [e.message for e in events if e.kind == "step"]
    assert any("CRAG active retrieval" in m for m in steps)
    done = events[-1]
    assert done.kind == "done" and done.documents == [web_doc]
    assert done.answer == "Every 30 days [1]."


def test_self_rag_regenerates_an_ungrounded_answer(config, embeddings):
    config.generation.self_rag.enabled = True
    pipeline = make_pipeline(
        config, embeddings,
        llm_responses=[
            "It weighs 9 tons.",
            '{"grounded": "no", "answers_question": "yes"}',
            "It weighs 340 grams [1].",
            '{"grounded": "yes", "answers_question": "yes"}',
        ],
    )
    pipeline.ingest()
    events = list(pipeline.run("How heavy is the Zorblax-9?"))
    assert [e.kind for e in events].count("reset") == 1
    assert events[-1].answer == "It weighs 340 grams [1]."


def test_parent_documents_replace_child_chunks(config, embeddings):
    pipeline = make_pipeline(config, embeddings)
    pipeline.ingest()
    docs = pipeline.retrieve("Zorblax-9 filter")
    # children are 40 tokens; the parent holds the whole short file
    assert any("12 volt" in d.page_content and "isopropyl" in d.page_content for d in docs)


def test_full_width_citations_are_normalized():
    from src.generation.generator import normalize_citations
    assert normalize_citations("Mistral 7B【1】") == "Mistral 7B[1]"


def test_lexical_semantic_splitter_cuts_at_the_topic_change():
    from src.config import ChunkingConfig
    from src.indexing.chunking import split_documents

    bread = " ".join(f"Sourdough bread needs flour, water, salt and a lively starter, step {i}." for i in range(8))
    orbit = " ".join(f"Satellites orbit Earth because gravity bends their path around the planet, fact {i}." for i in range(8))
    doc = Document(page_content=f"{bread} {orbit}", metadata={"source": "mixed.txt"})
    cfg = ChunkingConfig(chunk_size=400, chunk_overlap=0, method="semantic", semantic_boundaries="lexical")
    chunks = split_documents([doc], cfg, embeddings=DeterministicFakeEmbedding(size=8))
    assert len(chunks) == 2
    assert "Satellites" not in chunks[0].page_content and "Sourdough" not in chunks[1].page_content


def test_fast_pdf_loader_reads_pages_and_tables(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    from src.config import ExtractionConfig
    from src.indexing.loaders.fast_pdf import load_pdf

    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "The Zorblax-9 weighs 340 grams.")
    for y in (120, 140, 160):                       # a ruled 2x2 table
        page.draw_line((72, y), (272, y))
    for x in (72, 172, 272):
        page.draw_line((x, 120), (x, 160))
    page.insert_text((80, 135), "Model")
    page.insert_text((180, 135), "Grams")
    page.insert_text((80, 155), "Z9")
    page.insert_text((180, 155), "340")
    path = tmp_path / "w.pdf"
    pdf.save(path)

    docs = load_pdf(path, ExtractionConfig(ocr_enabled=False))
    assert docs[0].metadata == {"source": str(path), "page": 1, "type": "text"}
    assert "340 grams" in docs[0].page_content
    assert any(d.metadata["type"] == "table" and "Grams" in d.page_content for d in docs)
