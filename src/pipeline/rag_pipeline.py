"""
RAG Pipeline — Main Orchestrator

Ties together all components: document loading, chunking, embedding,
vector store, retrieval strategies, re-ranking, and generation.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from src.config import RAGConfig, load_config
from src.indexing.loaders.loader import load_documents, _load_from_file
from src.indexing.chunking import split_documents
from src.indexing.embeddings import create_embeddings, max_input_tokens
from src.indexing.multi_representation import (
    DocStore,
    SummaryWriter,
    build_representations,
    expand_to_parents,
)
from src.indexing.raptor import RAPTOR_TYPE, RaptorBuilder
from src.stores.vectorstore import (
    EmbeddingMismatchError,
    VectorStoreNotReadyError,
    add_documents,
    all_documents,
    backup_vectorstore,
    build_vectorstore,
    create_empty_vectorstore,
    delete_sources,
    load_vectorstore,
    read_stored_documents,
)
from src.stores.graph import GraphExtractor, GraphStore
from src.stores.relational import RelationalStore, extract_tables
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.active import QueryRewriter, web_search
from src.retrieval.ranking import Reranker
from src.retrieval.refinement import RelevanceGrader
from src.retrieval.utils import doc_key
from src.query_translation import create_translator
from src.query_translation.base import QueryTranslator
from src.query_translation.condense import QuestionCondenser
from src.routing.logical import GRAPH_DB, RELATIONAL_DB, VECTORSTORE, LogicalRouter
from src.routing.semantic import SemanticRouter
from src.query_construction.self_query import (
    MetadataCatalog,
    MetadataFilter,
    SelfQueryConstructor,
)
from src.query_construction.text_to_cypher import TextToCypher
from src.query_construction.text_to_sql import TextToSQL
from src.pipeline.planner import Plan, QueryPlanner
from src.generation.generator import (
    SOURCES_HEADER,
    create_llm,
    format_sources,
    stream_answer,
)
from src.generation.self_rag import AnswerGrader


# progress(stage, done, total) — stages: loading, tables, chunking, embedding, graph, summaries, done
Progress = Callable[[str, int, int], None]


@dataclass
class IngestResult:
    message: str
    written: int = 0
    loaded: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    # Set when ingest_files(background=True): runs graph extraction + RAPTOR.
    enrich: Callable[[Progress | None], None] | None = None


@dataclass
class Retrieval:
    search_query: str               # Standalone form of the question
    documents: list[Document]
    prompt_name: str = ""           # Chosen by semantic routing
    route_prompt: str = ""


@dataclass
class Event:
    """One thing that happened while answering, streamed to the UI and CLI."""
    kind: str                       # step | token | reset | done
    stage: str = ""                 # query_translation | routing | query_construction | retrieval | generation
    message: str = ""
    text: str = ""
    data: dict = field(default_factory=dict)
    answer: str = ""
    documents: list[Document] = field(default_factory=list)


class _Clock:
    def __init__(self):
        self._last = time.perf_counter()

    def lap(self) -> int:
        """Milliseconds since the previous lap."""
        now = time.perf_counter()
        ms, self._last = int((now - self._last) * 1000), now
        return ms


def _timed(fn: Callable, *args):
    """Call *fn* and return (result, elapsed milliseconds)."""
    start = time.perf_counter()
    result = fn(*args)
    return result, int((time.perf_counter() - start) * 1000)


def _drain(steps: Generator):
    """Exhaust a step generator and return its return value."""
    while True:
        try:
            next(steps)
        except StopIteration as stop:
            return stop.value


class RAGPipeline:
    """Main RAG pipeline orchestrator.

    Usage:
        # Ingest documents (the pipeline is ready to query afterwards)
        pipeline = RAGPipeline.from_config()
        pipeline.ingest()

        # Query an existing index
        pipeline = RAGPipeline.from_config()
        pipeline.load()
        answer, sources = pipeline.query("What is X?")
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._llm: BaseChatModel | None = None
        self._embeddings: Embeddings | None = None
        self._translator: QueryTranslator | None = None
        self._reranker: Reranker | None = None
        self._condenser: QuestionCondenser | None = None
        self._vectorstore = None
        self._base_retriever: HybridRetriever | None = None
        self.relational = RelationalStore(config.relational_db.path)
        self.graph = GraphStore(config.graph_db.path)
        self.docstore = DocStore(config.indexing.docstore_path)
        self._catalog = MetadataCatalog()
        self._router: LogicalRouter | None = None
        self._semantic_router: SemanticRouter | None = None
        self._relevance_grader: RelevanceGrader | None = None
        self._query_rewriter: QueryRewriter | None = None
        self._answer_grader: AnswerGrader | None = None
        self._planner: QueryPlanner | None = None
        self._plan_cache: OrderedDict = OrderedDict()
        self._cache_lock = threading.Lock()

    @classmethod
    def from_config(cls, config_path: str | None = None) -> "RAGPipeline":
        """Create a pipeline instance from a config file."""
        config = load_config(config_path)
        return cls(config)

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = create_llm(self.config.llm)
        return self._llm

    @property
    def embeddings(self) -> Embeddings:
        # Loading a local embedding model is slow; do it once per pipeline.
        if self._embeddings is None:
            self._embeddings = create_embeddings(self.config.embedding)
        return self._embeddings

    @property
    def is_ready(self) -> bool:
        return self._base_retriever is not None

    @property
    def strategy_name(self) -> str:
        if self._translator is None:
            return "N/A"
        name = self._translator.name
        if self._base_retriever is not None and self._base_retriever.uses_bm25:
            name += " + hybrid BM25"
        if self._reranker is not None and self._reranker.enabled:
            name += " + rerank"
        return name

    # -----------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------

    def ingest(self, rebuild: bool = False, progress: Progress | None = None) -> None:
        """Index the configured sources: load → chunk → embed → store → enrich.

        Chunks from the configured sources replace their previous versions, so
        re-running never duplicates them, while files uploaded through the UI
        are kept. With ``rebuild=True`` every store is wiped and rebuilt from
        the configured sources only. The pipeline is ready to query afterwards.
        """
        print("\n" + "=" * 60)
        print("  [IN] RAG Pipeline — Document Ingestion")
        print("=" * 60)
        started = time.perf_counter()

        documents = load_documents(self.config.documents, extraction_config=self.config.extraction)
        if not documents:
            print("\n[ERROR] No documents found. Check your config.yaml sources.")
            return

        written, enrich = self._index_documents(documents, rebuild=rebuild, progress=progress)
        if not written:
            print("\n[ERROR] No chunks generated. Check chunking settings.")
            return
        print(f"[OK] Searchable after {time.perf_counter() - started:.0f}s")
        enrich(progress)

        print("\n" + "=" * 60)
        print(f"  [OK] Ingestion complete in {time.perf_counter() - started:.0f}s")
        print("=" * 60 + "\n")

    def ingest_files(
        self,
        file_paths: list[str],
        progress: Progress | None = None,
        background: bool = False,
    ) -> IngestResult:
        """Index uploaded files.

        The files are searchable as soon as this returns. Graph extraction and
        RAPTOR summaries (the slow, LLM-bound part) run here too, unless
        *background* is true: then ``result.enrich(progress)`` is left for the
        caller to run (the web UI does it on a worker thread).
        """
        if not file_paths:
            return IngestResult("No files provided.")

        documents: list[Document] = []
        loaded, failed = [], []
        for i, fp in enumerate(file_paths):
            path = Path(fp)
            if not path.is_file():
                failed.append(path.name)
                continue
            if progress:
                progress("loading", i, len(file_paths))
            pages = (lambda done, total: progress("loading", done, total)) if progress else None
            docs = _load_from_file(path, extraction_config=self.config.extraction, progress=pages)
            if docs:
                documents.extend(docs)
                loaded.append(path.name)
            else:
                failed.append(path.name)

        if not documents:
            return IngestResult(
                f"Could not read {', '.join(failed)}. The file may be empty, scanned without OCR, or unsupported.",
                failed=failed,
            )

        written, enrich = self._index_documents(documents, progress=progress)
        message = f"Indexed {len(loaded)} file(s) into {written} chunks: {', '.join(loaded)}"
        if failed:
            message += f". Could not read: {', '.join(failed)}"
        result = IngestResult(message, written=written, loaded=loaded, failed=failed,
                              sources=sorted({str(d.metadata.get("source", "")) for d in documents}))
        if background:
            result.enrich = enrich
        else:
            enrich(progress)
        return result

    def _index_documents(
        self,
        documents: list[Document],
        rebuild: bool = False,
        progress: Progress | None = None,
    ) -> tuple[int, Callable[[Progress | None], None]]:
        """Make *documents* searchable; return (units written, enrich function).

        Tables go to the relational DB, parents to the docstore and retrieval
        units to the vectorstore. The returned function runs the LLM-bound
        enrichment (graph DB triples and RAPTOR summaries) for these sources.
        """
        sources = {str(d.metadata.get("source", "")) for d in documents}
        report = progress or (lambda *a: None)

        # Tables → relational DB
        report("tables", 0, 1)
        self._index_tables(documents, rebuild=rebuild)

        # Chunk optimization + multi-representation: units → vectorstore, parents → docstore
        report("chunking", 0, 1)
        units, parents = self._representations(documents)
        if not units:
            return 0, lambda _progress=None: None
        if rebuild:
            self.docstore.reset()
        self.docstore.delete_sources(sources)
        self.docstore.put(parents)

        embedding = lambda done, total: report("embedding", done, total)  # noqa: E731
        if rebuild:
            self._vectorstore = build_vectorstore(
                self.config.vectorstore, self.embeddings, self.config.embedding.signature,
                documents=[],
            )
        else:
            self._open_for_writing()
            delete_sources(self._vectorstore, sources)
        written = add_documents(self._vectorstore, units, progress=embedding)
        print(f"[OK] Indexed {written} chunks ({self._vectorstore._collection.count()} in the vector store)")
        self._init_retrieval()

        graph_input = list(parents.values()) or units

        def enrich(enrich_progress: Progress | None = None) -> None:
            self._enrich(sources, graph_input, rebuild, enrich_progress)

        return written, enrich

    def _enrich(
        self,
        sources: set[str],
        graph_input: list[Document],
        rebuild: bool = False,
        progress: Progress | None = None,
    ) -> None:
        """LLM-bound enrichment: graph DB triples, then RAPTOR summary trees."""
        started = time.perf_counter()
        self._index_graph(graph_input, sources, rebuild=rebuild, progress=progress)
        self._build_raptor(sources, progress=progress)
        self._init_retrieval()
        if progress:
            progress("done", 1, 1)
        print(f"[OK] Enrichment finished in {time.perf_counter() - started:.0f}s")

    def remove_source(self, source: str) -> None:
        """Delete everything indexed from *source* in every store."""
        if self._vectorstore is not None:
            delete_sources(self._vectorstore, {source})
        self.docstore.delete_sources({source})
        self.relational.delete_sources({source})
        self.graph.delete_sources({source})
        self.graph.save()
        if self._vectorstore is not None:
            self._init_retrieval()

    def list_sources(self) -> list[dict]:
        """Every indexed file with its chunk, table and summary counts."""
        if self._vectorstore is None:
            return []
        metas = self._vectorstore._collection.get(include=["metadatas"])["metadatas"]
        files: dict[str, dict] = {}
        for meta in metas:
            meta = meta or {}
            source = str(meta.get("source", ""))
            entry = files.setdefault(source, {"source": source, "name": source if source.startswith("http")
                                              else Path(source).name, "chunks": 0, "summaries": 0, "pages": 0})
            if meta.get("type") in (RAPTOR_TYPE, "summary"):
                entry["summaries"] += 1
            else:
                entry["chunks"] += 1
            if isinstance(meta.get("page"), int):
                entry["pages"] = max(entry["pages"], meta["page"])
        return sorted(files.values(), key=lambda f: f["name"].lower())

    def _representations(self, documents: list[Document]) -> tuple[list[Document], dict[str, Document]]:
        indexing = self.config.indexing
        summarizer = SummaryWriter(self.llm) if indexing.multi_representation == "summary" else None
        return build_representations(
            documents,
            indexing.multi_representation,
            split_parents=lambda docs: split_documents(
                docs, self.config.chunking.model_copy(
                    update={"chunk_size": indexing.parent_chunk_size, "method": "recursive"}
                ),
            ),
            split_children=self._split,
            summarizer=summarizer,
        )

    def _build_raptor(self, sources: set[str], progress: Progress | None = None) -> None:
        """Build a RAPTOR summary tree over each newly indexed source's chunks."""
        cfg = self.config.indexing.raptor
        if not cfg.enabled or self._vectorstore is None:
            return
        builder = RaptorBuilder(self.llm, self.embeddings, cfg.cluster_size, cfg.max_levels,
                                max_workers=self.config.indexing.llm_concurrency)
        collection = self._vectorstore._collection
        for source in sorted(sources):
            # Old summaries of this source went with delete_sources(); read its leaves back.
            data = collection.get(where={"source": source}, include=["documents", "metadatas", "embeddings"])
            leaves, vectors = [], []
            for text, meta, vector in zip(data["documents"], data["metadatas"], data["embeddings"]):
                if (meta or {}).get("type") not in (RAPTOR_TYPE, "summary"):
                    leaves.append(Document(page_content=text, metadata=meta or {}))
                    vectors.append(list(vector))
            if len(leaves) < 3:
                continue
            print(f"[RAPTOR] Building the summary tree for {Path(source).name} ({len(leaves)} chunks)...")
            step = (lambda done, total: progress("summaries", done, total)) if progress else None
            summaries = builder.build(leaves, vectors, source=source, progress=step)
            if summaries:
                add_documents(self._vectorstore, summaries)

    def _index_tables(self, documents: list[Document], rebuild: bool = False) -> None:
        """Load every table in *documents* into the relational DB (for Text-to-SQL)."""
        if not self.config.relational_db.enabled:
            return
        if rebuild:
            self.relational.reset()
        self.relational.delete_sources({str(d.metadata.get("source", "")) for d in documents})
        names = self.relational.add_tables(extract_tables(documents))
        if names:
            print(f"[SQL] Loaded {len(names)} table(s) into the relational DB: {', '.join(names)}")

    def _index_graph(
        self,
        chunks: list[Document],
        sources: set[str],
        rebuild: bool = False,
        progress: Progress | None = None,
    ) -> None:
        """Extract entity-relationship triples from *chunks* into the graph DB."""
        cfg = self.config.graph_db
        if not cfg.enabled:
            return
        print(f"[GRAPH] Extracting entities and relationships from {len(chunks)} chunk(s)...")
        extractor = GraphExtractor(self.llm, cfg.extraction_batch_chars, self.config.indexing.llm_concurrency)
        step = (lambda done, total: progress("graph", done, total)) if progress else None
        extracted = extractor.extract(chunks, progress=step)
        # Swap the new facts in at once, so queries never see a half-built graph.
        if rebuild:
            self.graph.reset()
        self.graph.replace_sources(sources, extracted)
        self.graph.save()
        print(f"[GRAPH] {self.graph.graph.number_of_edges()} relationship(s) in the graph DB")

    def _open_for_writing(self) -> None:
        """Open the index for adding chunks, creating or migrating it if needed."""
        if self._vectorstore is not None:
            return
        signature = self.config.embedding.signature
        try:
            self._vectorstore = load_vectorstore(
                self.config.vectorstore, self.embeddings, signature,
            )
        except EmbeddingMismatchError as e:
            print(f"[INFO] {e}")
            self._reembed_existing()
        except VectorStoreNotReadyError:
            self._vectorstore = create_empty_vectorstore(
                self.config.vectorstore, self.embeddings, signature,
            )

    def _reembed_existing(self) -> None:
        """Re-embed the chunks already in the index with the current model.

        The stored text may be the only copy left (e.g. of files uploaded
        through the UI), so it is migrated rather than discarded, after
        backing up the old index.
        """
        backup = backup_vectorstore(self.config.vectorstore)
        print(f"[SAVE] Backed up the existing index to: {backup}")

        old_chunks = read_stored_documents(self.config.vectorstore)
        print(f"[EMB] Re-embedding {len(old_chunks)} stored chunks with "
              f"{self.config.embedding.signature}...")
        self._vectorstore = build_vectorstore(
            self.config.vectorstore,
            self.embeddings,
            self.config.embedding.signature,
            documents=old_chunks,
        )

    def _split(self, documents: list[Document]) -> list[Document]:
        chunking = self.config.chunking
        limit = max_input_tokens(self.embeddings)
        approx_tokens = chunking.chunk_size if chunking.use_tiktoken else chunking.chunk_size // 4
        if limit and approx_tokens > limit:
            print(
                f"[WARN]  chunk_size ({chunking.chunk_size}) exceeds what the embedding "
                f"model reads (~{limit} tokens); the rest of each chunk will be ignored "
                "when searching. Lower chunking.chunk_size."
            )
        return split_documents(documents, chunking, self.embeddings)

    # -----------------------------------------------------------------
    # Loading (for query mode)
    # -----------------------------------------------------------------

    def load(self) -> None:
        """Load existing vector store and initialize retrieval components.

        Raises:
            VectorStoreNotReadyError: if there is no usable index (missing,
                empty, or built with a different embedding model).
        """
        print("\n[RANK] Loading RAG pipeline...")

        self._vectorstore = load_vectorstore(
            self.config.vectorstore,
            self.embeddings,
            self.config.embedding.signature,
        )
        self._init_retrieval()

        print(f"[STRAT] Retrieval strategy: {self.strategy_name}")
        print("[OK] Pipeline ready!\n")

    def warm_up(self) -> None:
        """Load models and caches now, so the first question isn't the slow one."""
        started = time.perf_counter()
        self.embeddings.embed_query("warm up")
        if self._reranker is not None:
            self._reranker.warm_up()
        self._semantic_route("warm up")
        _ = self.llm  # build the client
        print(f"[OK] Warmed up in {time.perf_counter() - started:.1f}s")

    def _llm_if_needed(self):
        return self.llm if self.config.reranking.provider == "rankgpt" else None

    def _init_retrieval(self) -> None:
        retrieval = self.config.retrieval

        if self._reranker is None:
            self._reranker = Reranker(self.config.reranking, llm=self._llm_if_needed())

        # When reranking, give it a wider candidate pool to choose from.
        per_query_k = (
            max(retrieval.top_k, self.config.reranking.candidates)
            if self._reranker.enabled
            else retrieval.top_k
        )

        hybrid = retrieval.hybrid
        stored = all_documents(self._vectorstore)
        self._catalog = MetadataCatalog.from_documents(stored)
        self._base_retriever = HybridRetriever(
            self._vectorstore,
            documents=stored if hybrid.enabled else [],
            k=per_query_k,
            fetch_k=hybrid.fetch_k,
            rrf_k=hybrid.rrf_k,
            use_bm25=hybrid.enabled,
        )
        self._translator = create_translator(retrieval, self.llm)

    @staticmethod
    def _step(stage: str, message: str, **data) -> Event:
        """A pipeline step, logged to the console and yielded to the caller (UI/CLI)."""
        print(f"[{stage.upper()}] {message}")
        return Event("step", stage=stage, message=message, data=data)

    # -----------------------------------------------------------------
    # Query
    # -----------------------------------------------------------------

    def available_sources(self) -> dict[str, str]:
        """Data stores that hold data, with the description the router sees."""
        sources = {VECTORSTORE: "text passages from the indexed documents"}
        qc = self.config.query_construction
        if self.config.relational_db.enabled and qc.text_to_sql:
            tables = self.relational.table_names()
            if tables:
                sources[RELATIONAL_DB] = f"SQL tables extracted from the documents: {', '.join(tables)}"
        if self.config.graph_db.enabled and qc.text_to_cypher and not self.graph.is_empty:
            sources[GRAPH_DB] = (
                f"knowledge graph of {self.graph.graph.number_of_nodes()} entities and "
                "the relationships between them"
            )
        return sources

    def _condense(self, question: str, chat_history: str) -> Iterator[Event]:
        """Make a follow-up standalone so every later step sees the full intent."""
        if not chat_history or not self.config.chat.rewrite_followups:
            return question
        if self._condenser is None:
            self._condenser = QuestionCondenser(self.llm)
        search_query = self._condenser.condense(question, chat_history)
        if search_query != question:
            yield self._step("query_translation", f"Rewrote follow-up as: {search_query}")
        return search_query

    def _num_queries(self) -> int:
        r = self.config.retrieval
        return {"multi_query": r.multi_query.num_queries, "rag_fusion": r.rag_fusion.num_queries}.get(
            r.strategy, 3)

    def _semantic_route(self, question: str) -> tuple[str, str, float]:
        cfg = self.config.routing
        if not cfg.semantic or not cfg.prompts:
            return "", "", 0.0
        if self._semantic_router is None:
            self._semantic_router = SemanticRouter(self.embeddings, {
                name: (r.description, r.prompt) for name, r in cfg.prompts.items()
            })
        return self._semantic_router.route(question)

    def _plan(self, search_query: str, available: dict[str, str]) -> Plan:
        """Query translation + logical routing + self-query: one LLM call, or one each."""
        cfg = self.config
        use_filter = cfg.query_construction.self_query
        routes = available if cfg.routing.logical else {VECTORSTORE: available[VECTORSTORE]}

        if cfg.retrieval.planner:
            key = (search_query, tuple(available))
            with self._cache_lock:
                if key in self._plan_cache:
                    self._plan_cache.move_to_end(key)
                    return self._plan_cache[key]
            if self._planner is None:
                self._planner = QueryPlanner(self.llm, cfg.retrieval.strategy, self._num_queries())
            plan = self._planner.plan(search_query, self._translator, routes, self._catalog, use_filter)
            with self._cache_lock:
                self._plan_cache[key] = plan
                if len(self._plan_cache) > 128:
                    self._plan_cache.popitem(last=False)
        else:
            queries = self._translator.translate(search_query)
            if len(routes) > 1:
                if self._router is None:
                    self._router = LogicalRouter(self.llm)
                datasources = self._router.route(search_query, routes)
            else:
                datasources = list(routes)
            metadata_filter = (
                SelfQueryConstructor(self.llm).construct(search_query, self._catalog)
                if use_filter and self._catalog.filterable else MetadataFilter()
            )
            plan = Plan(queries, datasources, metadata_filter)

        if not cfg.routing.logical:
            plan = Plan(plan.queries, list(available), plan.metadata_filter, plan.planned)
        return plan

    def _retrieve(self, search_query: str) -> Iterator[Event]:
        """Translation → routing → query construction → stores → ranking → CRAG.

        A generator of step events whose return value is the Retrieval.
        """
        if self._base_retriever is None:
            raise RuntimeError("Pipeline not loaded. Call .load() or .ingest() first.")
        cfg = self.config
        available = self.available_sources()

        # 1-2. Query translation + routing (+ the self-query filter), with the
        # semantic-routing embedding computed while the planner call is out.
        clock = _Clock()
        with ThreadPoolExecutor(max_workers=2) as pool:
            semantic = pool.submit(self._semantic_route, search_query)
            plan = self._plan(search_query, available)
            prompt_name, route_prompt, score = semantic.result()
        planning_ms = clock.lap()
        how = "one planner call" if plan.planned and cfg.retrieval.planner else "no LLM call needed"
        if not cfg.retrieval.planner:
            how = "separate calls"
        noun = "query" if len(plan.queries) == 1 else "queries"
        yield self._step("query_translation",
                         f"{self._translator.name}: {len(plan.queries)} {noun} ({how})",
                         queries=plan.queries, ms=planning_ms)
        yield self._step("routing", f"Logical routing → {', '.join(plan.datasources)}",
                         datasources=plan.datasources)
        if prompt_name:
            yield self._step("routing", f"Semantic routing → {prompt_name} prompt (similarity {score:.2f})",
                             prompt=prompt_name)

        # 3. Query construction + store lookups, all stores in parallel
        structured: list[Document] = []
        passages: list[Document] = []
        steps: list[Event] = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = {}
            if RELATIONAL_DB in plan.datasources:
                jobs[RELATIONAL_DB] = pool.submit(_timed, TextToSQL(
                    self.llm, self.relational, cfg.relational_db.max_rows).run, search_query)
            if GRAPH_DB in plan.datasources:
                jobs[GRAPH_DB] = pool.submit(_timed, TextToCypher(
                    self.llm, self.graph, cfg.graph_db.max_results).run, search_query)
            if VECTORSTORE in plan.datasources:
                jobs[VECTORSTORE] = pool.submit(_timed, self._search, plan.queries, plan.metadata_filter)

            if RELATIONAL_DB in jobs:
                (sql, docs), ms = jobs[RELATIONAL_DB].result()
                structured += docs
                steps.append(self._step(
                    "query_construction",
                    f"Text-to-SQL: {sql}" if docs else "Text-to-SQL found no answer in the tables",
                    sql=sql, store=RELATIONAL_DB, ms=ms))
            if GRAPH_DB in jobs:
                (cypher, docs), ms = jobs[GRAPH_DB].result()
                structured += docs
                if docs:
                    message = f"Text-to-Cypher: {cypher}" if cypher else "Graph DB: facts around the entities mentioned"
                else:
                    message = "Text-to-Cypher found nothing in the graph"
                steps.append(self._step("query_construction", message, cypher=cypher, store=GRAPH_DB, ms=ms))
            if VECTORSTORE in jobs:
                passages, ms = jobs[VECTORSTORE].result()
        for step in steps:
            yield step
        if VECTORSTORE not in jobs and not structured:
            # The routed stores had nothing: fall back to the vectorstore.
            passages, ms = _timed(self._search, plan.queries, plan.metadata_filter)
        if VECTORSTORE in jobs or not structured:
            label = f"Self-query filter: {plan.metadata_filter.describe()} · " if plan.metadata_filter else ""
            yield self._step("query_construction", f"{label}Vectorstore: {len(passages)} candidate passage(s)",
                             store=VECTORSTORE, ms=ms)

        # 4. Retrieval: ranking
        clock.lap()
        passages = self._rerank(search_query, passages)
        if self._reranker is not None and self._reranker.enabled and passages:
            yield self._step("retrieval", f"Re-ranked with {cfg.reranking.provider} → best {len(passages)}",
                             ms=clock.lap())

        # 5. Retrieval: refinement and active retrieval (CRAG)
        crag = cfg.retrieval.crag
        if crag.enabled and (structured or passages):
            total = len(structured) + len(passages)
            graded = {doc_key(d) for d in passages}
            structured, passages = self._grade(search_query, structured, passages)
            relevant = len(structured) + len(passages)
            yield self._step("retrieval", f"CRAG: {relevant}/{total} relevant ({self._grading_mode()})",
                             ms=clock.lap())
            if relevant < crag.min_relevant:
                new_query = self._rewriter().rewrite(search_query)
                where = "the vectorstore" + (" and the web" if crag.web_search else "")
                yield self._step("retrieval", f"CRAG active retrieval: searching {where} for: {new_query}")
                extra = [d for d in self._search([new_query]) if doc_key(d) not in graded]
                if crag.web_search:
                    extra += web_search(new_query, crag.web_results)
                extra = self._rerank(search_query, extra)
                _, found = self._grade(search_query, [], extra)
                yield self._step("retrieval", f"CRAG: {len(found)}/{len(extra)} new document(s) relevant",
                                 ms=clock.lap())
                passages += found

        documents = structured + self._expand(passages)
        yield self._step("retrieval", f"{len(documents)} document(s) passed to the LLM")
        return Retrieval(search_query, documents, prompt_name, route_prompt)

    def _search(self, queries: list[str], metadata_filter=None) -> list[Document]:
        """Vectorstore search for every query, fused by the translator."""
        results = [self._base_retriever.invoke(q, metadata_filter) for q in queries]
        if metadata_filter and not any(results):
            print("[INFO] The metadata filter matched nothing; searching everything.")
            results = [self._base_retriever.invoke(q) for q in queries]
        return self._translator.fuse(results)

    def _rerank(self, query: str, passages: list[Document]) -> list[Document]:
        """Re-rank (or cap) the candidates; scores land in ``relevance_score``."""
        if self._reranker is not None and self._reranker.enabled and passages:
            return self._reranker.rerank(query, passages[: self.config.reranking.candidates])
        return passages[: self.config.retrieval.top_k]

    def _expand(self, passages: list[Document]) -> list[Document]:
        """Swap retrieval units for their parent documents (multi-representation)."""
        if self.config.indexing.multi_representation != "none":
            return expand_to_parents(passages, self.docstore)
        return passages

    def _rank(self, query: str, passages: list[Document]) -> list[Document]:
        return self._expand(self._rerank(query, passages))

    def _grading_mode(self) -> str:
        crag = self.config.retrieval.crag
        if crag.grader == "reranker" and self._reranker is not None and self._reranker.scores_relevance:
            return "re-ranker scores"
        return "LLM grader"

    def _grade(
        self, query: str, structured: list[Document], passages: list[Document],
    ) -> tuple[list[Document], list[Document]]:
        """CRAG refinement: keep only the relevant documents.

        With a cross-encoder re-ranker its scores already measure relevance,
        so passages are graded by threshold with no LLM call (the CRAG paper's
        evaluator is a small model too). Otherwise the LLM grades everything.
        """
        crag = self.config.retrieval.crag
        if self._grading_mode() == "re-ranker scores":
            kept = [p for p in passages
                    if p.metadata.get("relevance_score", crag.min_score) >= crag.min_score]
            return structured, kept
        expanded = self._expand(passages)
        verdicts = self._grader().grade(query, structured + expanded)
        keep = [d for d, ok in zip(structured + expanded, verdicts) if ok]
        return [d for d in keep if d in structured], [d for d in keep if d not in structured]

    def _grader(self) -> RelevanceGrader:
        if self._relevance_grader is None:
            self._relevance_grader = RelevanceGrader(self.llm)
        return self._relevance_grader

    def _rewriter(self) -> QueryRewriter:
        if self._query_rewriter is None:
            self._query_rewriter = QueryRewriter(self.llm)
        return self._query_rewriter

    def retrieve(self, question: str, chat_history: str = "") -> list[Document]:
        """Find the documents to answer *question* with."""
        def steps():
            search_query = yield from self._condense(question, chat_history)
            return (yield from self._retrieve(search_query))
        return _drain(steps()).documents

    def _system_prompt(self, retrieval: Retrieval) -> str:
        prompt = self.config.chat.system_prompt
        return f"{prompt.strip()}\n\n{retrieval.route_prompt}" if retrieval.route_prompt else prompt

    def run(self, question: str, chat_history: str = "") -> Iterator[Event]:
        """Run the whole pipeline, yielding events as it goes.

        Events: ``step`` (stage + message), ``token`` (answer text),
        ``reset`` (Self-RAG discarded the answer streamed so far) and a final
        ``done`` carrying the answer and its source documents.
        """
        started = time.perf_counter()
        search_query = yield from self._condense(question, chat_history)
        retrieval = yield from self._retrieve(search_query)
        first_token_ms = None

        self_rag = self.config.generation.self_rag
        extra_rule = ""
        answer = ""
        for attempt in range(1 + (self_rag.max_retries if self_rag.enabled else 0)):
            answer = ""
            for piece in stream_answer(
                llm=self.llm,
                question=question,
                documents=retrieval.documents,
                chat_history=chat_history,
                system_prompt=self._system_prompt(retrieval) + extra_rule,
            ):
                if not piece:   # reasoning models stream empty chunks while thinking
                    continue
                if first_token_ms is None:
                    first_token_ms = int((time.perf_counter() - started) * 1000)
                answer += piece
                yield Event("token", text=piece)

            if not self_rag.enabled:
                break
            if self._answer_grader is None:
                self._answer_grader = AnswerGrader(self.llm)
            check = self._answer_grader.check(question, retrieval.documents, answer)
            yield self._step(
                "generation",
                f"Self-RAG: grounded={'yes' if check.grounded else 'no'}, "
                f"answers the question={'yes' if check.answers_question else 'no'}",
            )
            if check.ok or attempt == self_rag.max_retries:
                break

            yield Event("reset")
            if not check.answers_question:
                # RRR: rewrite the question and re-retrieve before reading again
                new_query = self._rewriter().rewrite(
                    retrieval.search_query, hint="The previous search did not find what was needed.",
                )
                yield self._step("generation", f"Self-RAG re-retrieval with: {new_query}")
                retrieval = yield from self._retrieve(new_query)
            else:
                yield self._step("generation", "Self-RAG: regenerating a strictly grounded answer")
                extra_rule = (
                    "\n\nYour previous answer contained claims the passages do not support. "
                    "State only what the passages say."
                )

        yield Event("done", answer=answer, documents=retrieval.documents, data={
            "ms": int((time.perf_counter() - started) * 1000),
            "first_token_ms": first_token_ms,
            "model": getattr(self.llm, "last_used", "") or f"{self.config.llm.provider}:{self.config.llm.model}",
        })

    def query(
        self,
        question: str,
        chat_history: str = "",
    ) -> tuple[str, list[Document]]:
        """Run the full RAG query pipeline and return (answer, source_documents)."""
        done = [e for e in self.run(question, chat_history) if e.kind == "done"][-1]
        return done.answer, done.documents

    def format_response(
        self, answer: str, documents: list[Document]
    ) -> str:
        """Format a response with optional source citations."""
        response = answer

        if self.config.chat.show_sources and documents:
            sources = format_sources(documents)
            if sources:
                response += f"\n\n{SOURCES_HEADER}\n{sources}"

        return response
