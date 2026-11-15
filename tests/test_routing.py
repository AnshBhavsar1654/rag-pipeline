"""Offline tests for the graph DB, routing and self-query."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import FakeListChatModel

from src.query_construction.self_query import MetadataCatalog, SelfQueryConstructor
from src.query_construction.text_to_cypher import TextToCypher
from src.routing.logical import LogicalRouter
from src.routing.semantic import SemanticRouter
from src.stores.graph import GraphExtractor, GraphStore, parse_triples


TRIPLES = """Here you go:
[{"head": "GPT-4", "head_type": "Model", "relation": "developed by", "tail": "OpenAI", "tail_type": "Organization"},
 {"head": "GPT-4", "head_type": "Model", "relation": "USES", "tail": "Transformer", "tail_type": "Architecture"}]"""


def make_graph(tmp_path) -> GraphStore:
    store = GraphStore(tmp_path / "graph.json")
    store.add_triples(parse_triples(TRIPLES), source="llm.md")
    store.save()
    return store


def test_graph_persists_and_answers_cypher(tmp_path):
    make_graph(tmp_path)
    store = GraphStore(tmp_path / "graph.json")  # reload from disk
    rows = store.run_cypher(
        'MATCH (m:Model)-[r:DEVELOPED_BY]->(o) WHERE m.name == "gpt-4" RETURN o.name'
    )
    assert rows == [{"o.name": "openai"}]
    assert "DEVELOPED_BY" in store.schema()


def test_graph_delete_sources_removes_orphans(tmp_path):
    store = make_graph(tmp_path)
    store.delete_sources({"llm.md"})
    assert store.is_empty and store.graph.number_of_nodes() == 0


def test_graph_extractor_batches_chunks(tmp_path):
    store = GraphStore(tmp_path / "g.json")
    chunks = [Document(page_content=f"chunk {i}", metadata={"source": "a.md"}) for i in range(3)]
    added = GraphExtractor(FakeListChatModel(responses=[TRIPLES, "[]"]), batch_size=2).build(store, chunks)
    assert added == 2


def test_text_to_cypher_falls_back_to_entity_facts(tmp_path):
    store = make_graph(tmp_path)
    llm = FakeListChatModel(responses=["not cypher at all", "MATCH (a RETURN broken"])
    cypher, docs = TextToCypher(llm, store).run("Who made GPT-4?")
    assert cypher == ""
    assert "(gpt-4) -[DEVELOPED_BY]-> (openai)" in docs[0].page_content


def test_logical_router_parses_choices_in_order():
    available = {"vectorstore": "text", "relational_db": "tables", "graph_db": "graph"}
    router = LogicalRouter(FakeListChatModel(responses=["relational_db, vectorstore"]))
    assert router.route("total revenue?", available) == ["relational_db", "vectorstore"]
    assert LogicalRouter(FakeListChatModel(responses=["??"])).route("q", available) == ["vectorstore"]


class KeywordEmbeddings(Embeddings):
    """Tiny bag-of-words embedding so similarity is predictable."""

    VOCAB = ["compare", "versus", "total", "price", "explain", "how"]

    def _vec(self, text):
        words = text.lower().replace("?", "").split()
        return [float(words.count(w)) + 0.01 for w in self.VOCAB]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def test_semantic_router_picks_closest_prompt():
    router = SemanticRouter(KeywordEmbeddings(), {
        "compare": ("compare versus", "Use a table."),
        "numbers": ("total price", "Lead with figures."),
        "explain": ("explain how", "Go step by step."),
    })
    name, prompt, _ = router.route("compare GPT versus Llama")
    assert (name, prompt) == ("compare", "Use a table.")
    assert router.route("what is the total price")[0] == "numbers"


def test_self_query_keeps_only_known_values():
    docs = [
        Document(page_content="a", metadata={"source": "/x/report.pdf", "type": "text", "page": 1}),
        Document(page_content="b", metadata={"source": "/x/notes.md", "type": "table", "page": 4}),
    ]
    catalog = MetadataCatalog.from_documents(docs)
    llm = FakeListChatModel(responses=['{"source": ["REPORT.pdf", "ghost.pdf"], "type": "table", "page": [99]}'])
    flt = SelfQueryConstructor(llm).construct("tables in the report", catalog)
    assert flt.conditions == {"source": ["/x/report.pdf"], "type": ["table"]}
    assert flt.to_chroma() == {"$and": [{"source": "/x/report.pdf"}, {"type": "table"}]}
    assert flt.matches(docs[0].metadata) is False


def test_rankgpt_ranking_parser_appends_missing():
    from src.retrieval.ranking import parse_ranking
    assert parse_ranking("[3] > [1]", 4) == [2, 0, 1, 3]


def test_relevance_parser():
    from src.retrieval.refinement import parse_relevant
    assert parse_relevant("1, 3", 3) == [True, False, True]
    assert parse_relevant("none", 2) == [False, False]


def test_raptor_builds_levels_until_one_summary():
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from src.indexing.raptor import RaptorBuilder

    leaves = [Document(page_content=f"passage {i}") for i in range(10)]
    llm = FakeListChatModel(responses=["summary"])
    summaries = RaptorBuilder(llm, DeterministicFakeEmbedding(size=8), cluster_size=4).build(leaves, source="a.md")
    assert {d.metadata["source"] for d in summaries} == {"a.md"}
    levels = [d.metadata["raptor_level"] for d in summaries]
    assert levels[0] == 1 and levels[-1] == max(levels) and levels.count(max(levels)) == 1


def test_triples_survive_a_reply_cut_off_mid_json():
    cut = TRIPLES[: TRIPLES.rindex('"tail"')]  # second triple truncated
    assert [t["tail"] for t in parse_triples(cut)] == ["OpenAI"]
