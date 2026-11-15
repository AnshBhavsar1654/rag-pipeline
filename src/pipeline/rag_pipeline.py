"""
RAG Pipeline — Main Orchestrator

Ties together all components: document loading, chunking, embedding,
vector store, retrieval strategies, re-ranking, and generation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
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
from src.retrieval.ranking import Reranker
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
from src.generation.generator import (
    SOURCES_HEADER,
    create_llm,
    format_sources,
    generate_answer,
    stream_answer,
)


@dataclass
class Retrieval:
    question: str
    search_query: str               # Standalone form of the question
    documents: list[Document]
    prompt_name: str = ""           # Chosen by semantic routing
    route_prompt: str = ""


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
        # Receives (stage, message, data) for every pipeline step; set by the UI.
        self._listener: Callable[[str, str, dict], None] | None = None

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

    def ingest(self, rebuild: bool = False) -> None:
        """Full ingestion pipeline: load → chunk → embed → store.

        Chunks from the configured sources replace their previous versions, so
        re-running never duplicates them, while files uploaded through the UI
        are kept. With ``rebuild=True`` the index is wiped and rebuilt from the
        configured sources only. The pipeline is ready to query afterwards.
        """
        print("\n" + "=" * 60)
        print("  [IN] RAG Pipeline — Document Ingestion")
        print("=" * 60)

        # 1. Load documents (with table/image extraction)
        documents = load_documents(
            self.config.documents,
            extraction_config=self.config.extraction,
        )
        if not documents:
            print("\n[ERROR] No documents found. Check your config.yaml sources.")
            return

        if not self._index_documents(documents, rebuild=rebuild):
            print("\n[ERROR] No chunks generated. Check chunking settings.")
            return

        print("\n" + "=" * 60)
        print("  [OK] Ingestion complete!")
        print("=" * 60 + "\n")

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

    def ingest_files(self, file_paths: list[str]) -> str:
        """Ingest uploaded files into the existing vector store.

        Loads, chunks, embeds, and adds the documents to the running
        vector store so the chatbot can answer questions about them.
        Re-uploading an unchanged file does not create duplicate chunks.

        Returns a status message describing what happened.
        """
        if not file_paths:
            return "No files provided."

        # Load documents from each file
        all_docs: list[Document] = []
        loaded_names: list[str] = []
        failed_names: list[str] = []
        for fp in file_paths:
            path = Path(fp)
            if not path.is_file():
                failed_names.append(path.name)
                continue
            docs = _load_from_file(path, extraction_config=self.config.extraction)
            if docs:
                all_docs.extend(docs)
                loaded_names.append(path.name)
            else:
                failed_names.append(path.name)

        if not all_docs:
            return f"Failed to load any documents. Unsupported or unreadable: {', '.join(failed_names)}"

        written = self._index_documents(all_docs)

        msg = f"Successfully ingested {len(loaded_names)} file(s) ({written} chunks): {', '.join(loaded_names)}"
        if failed_names:
            msg += f"\nFailed: {', '.join(failed_names)}"
        return msg

    def _index_documents(self, documents: list[Document], rebuild: bool = False) -> int:
        """Index *documents* into every store; returns the vector units written.

        Chunks from the same sources replace their previous versions, so
        re-running never duplicates them. The pipeline is ready afterwards.
        """
        sources = {str(d.metadata.get("source", "")) for d in documents}

        # Tables → relational DB
        self._index_tables(documents, rebuild=rebuild)

        # Chunk optimization + multi-representation: units → vectorstore, parents → docstore
        units, parents = self._representations(documents)
        if not units:
            return 0
        if rebuild:
            self.docstore.reset()
        self.docstore.delete_sources(sources)
        self.docstore.put(parents)

        if rebuild:
            self._vectorstore = build_vectorstore(
                self.config.vectorstore,
                self.embeddings,
                self.config.embedding.signature,
                documents=units,
            )
            written = len(units)
        else:
            self._open_for_writing()
            delete_sources(self._vectorstore, sources)
            written = add_documents(self._vectorstore, units)
            total = self._vectorstore._collection.count()
            print(f"[OK] Indexed {written} chunks ({total} in the vector store)")

        # Entities and relationships → graph DB (parents: fewer, fuller passages)
        self._index_graph(list(parents.values()) or units, sources, rebuild=rebuild)

        # Hierarchical indexing: RAPTOR summary tree over everything indexed
        self._build_raptor()

        self._init_retrieval()
        return written

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

    def _build_raptor(self) -> None:
        """(Re)build the RAPTOR summary tree over every leaf chunk in the index."""
        cfg = self.config.indexing.raptor
        if not cfg.enabled or self._vectorstore is None:
            return
        collection = self._vectorstore._collection
        collection.delete(where={"type": RAPTOR_TYPE})
        data = collection.get(include=["documents", "metadatas", "embeddings"])
        leaves, vectors = [], []
        for text, meta, vector in zip(data["documents"], data["metadatas"], data["embeddings"]):
            if (meta or {}).get("type") not in (RAPTOR_TYPE, "summary"):
                leaves.append(Document(page_content=text, metadata=meta or {}))
                vectors.append(list(vector))
        if len(leaves) < 2:
            return
        print(f"[RAPTOR] Building the summary tree over {len(leaves)} chunk(s)...")
        builder = RaptorBuilder(self.llm, self.embeddings, cfg.cluster_size, cfg.max_levels)
        summaries = builder.build(leaves, vectors)
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

    def _index_graph(self, chunks: list[Document], sources: set[str], rebuild: bool = False) -> None:
        """Extract entity-relationship triples from *chunks* into the graph DB."""
        if not self.config.graph_db.enabled:
            return
        if rebuild:
            self.graph.reset()
        self.graph.delete_sources(sources)
        print(f"[GRAPH] Extracting entities and relationships from {len(chunks)} chunk(s)...")
        extractor = GraphExtractor(self.llm, self.config.graph_db.extraction_batch_size)
        added = extractor.build(self.graph, chunks)
        self.graph.save()
        print(f"[GRAPH] Added {added} relationship(s) "
              f"({self.graph.graph.number_of_edges()} in the graph DB)")

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

    def _init_retrieval(self) -> None:
        retrieval = self.config.retrieval

        if self._reranker is None:
            self._reranker = Reranker(self.config.reranking)

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

    def _emit(self, stage: str, message: str, **data) -> None:
        """Report a pipeline step to the console and to the active listener (UI)."""
        print(f"[{stage.upper()}] {message}")
        if self._listener is not None:
            self._listener(stage, message, data)

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

    def _retrieve(self, question: str, chat_history: str = "") -> Retrieval:
        """Question → translation → routing → query construction → stores → ranking."""
        if self._base_retriever is None:
            raise RuntimeError("Pipeline not loaded. Call .load() or .ingest() first.")
        cfg = self.config

        # Make follow-ups standalone so every later step sees the full intent
        search_query = question
        if chat_history and cfg.chat.rewrite_followups:
            if self._condenser is None:
                self._condenser = QuestionCondenser(self.llm)
            search_query = self._condenser.condense(question, chat_history)
            if search_query != question:
                self._emit("query_translation", f"Rewrote follow-up as: {search_query}")

        # 1. Query translation
        queries = self._translator.translate(search_query)
        noun = "query" if len(queries) == 1 else "queries"
        self._emit(
            "query_translation",
            f"{self._translator.name}: searching with {len(queries)} {noun}",
            queries=queries,
        )

        # 2. Routing
        available = self.available_sources()
        if cfg.routing.logical and len(available) > 1:
            if self._router is None:
                self._router = LogicalRouter(self.llm)
            datasources = self._router.route(search_query, available)
        else:
            datasources = list(available)
        self._emit("routing", f"Logical routing → {', '.join(datasources)}", datasources=datasources)

        prompt_name, route_prompt = "", ""
        if cfg.routing.semantic and cfg.routing.prompts:
            if self._semantic_router is None:
                self._semantic_router = SemanticRouter(self.embeddings, {
                    name: (r.description, r.prompt) for name, r in cfg.routing.prompts.items()
                })
            prompt_name, route_prompt, score = self._semantic_router.route(search_query)
            self._emit(
                "routing",
                f"Semantic routing → {prompt_name} prompt (similarity {score:.2f})",
                prompt=prompt_name,
            )

        # 3. Query construction, one per routed data store
        structured: list[Document] = []
        if RELATIONAL_DB in datasources:
            sql, docs = TextToSQL(self.llm, self.relational, cfg.relational_db.max_rows).run(search_query)
            self._emit(
                "query_construction",
                f"Text-to-SQL: {sql}" if docs else "Text-to-SQL found no answer in the tables",
                sql=sql,
            )
            structured += docs
        if GRAPH_DB in datasources:
            cypher, docs = TextToCypher(self.llm, self.graph, cfg.graph_db.max_results).run(search_query)
            if docs:
                message = f"Text-to-Cypher: {cypher}" if cypher else "Graph DB: facts around the entities mentioned"
            else:
                message = "Text-to-Cypher found nothing in the graph"
            self._emit("query_construction", message, cypher=cypher)
            structured += docs

        passages: list[Document] = []
        if VECTORSTORE in datasources or not structured:
            metadata_filter = MetadataFilter()
            if cfg.query_construction.self_query and self._catalog.filterable:
                metadata_filter = SelfQueryConstructor(self.llm).construct(search_query, self._catalog)
                self._emit(
                    "query_construction",
                    f"Self-query filter: {metadata_filter.describe()}" if metadata_filter
                    else "Self-query: no metadata filter needed",
                )
            passages = self._search(queries, metadata_filter)

        # 4. Retrieval: rank the passages, keep the best
        if self._reranker is not None and self._reranker.enabled and passages:
            passages = self._reranker.rerank(search_query, passages[: cfg.reranking.candidates])
            self._emit("retrieval", f"Re-ranked with {cfg.reranking.provider} → kept {len(passages)}")
        else:
            passages = passages[: cfg.retrieval.top_k]
        if cfg.indexing.multi_representation != "none":
            passages = expand_to_parents(passages, self.docstore)

        documents = structured + passages
        self._emit("retrieval", f"{len(documents)} document(s) retrieved")
        return Retrieval(question, search_query, documents, prompt_name, route_prompt)

    def _search(self, queries: list[str], metadata_filter=None) -> list[Document]:
        """Vectorstore search for every query, fused by the translator."""
        results = [self._base_retriever.invoke(q, metadata_filter) for q in queries]
        if metadata_filter and not any(results):
            print("[INFO] The metadata filter matched nothing; searching everything.")
            results = [self._base_retriever.invoke(q) for q in queries]
        return self._translator.fuse(results)

    def retrieve(self, question: str, chat_history: str = "") -> list[Document]:
        """Find the documents to answer *question* with."""
        return self._retrieve(question, chat_history).documents

    def _system_prompt(self, retrieval: Retrieval) -> str:
        prompt = self.config.chat.system_prompt
        return f"{prompt.strip()}\n\n{retrieval.route_prompt}" if retrieval.route_prompt else prompt

    def query(
        self,
        question: str,
        chat_history: str = "",
    ) -> tuple[str, list[Document]]:
        """Run the full RAG query pipeline and return (answer, source_documents)."""
        retrieval = self._retrieve(question, chat_history)
        answer = generate_answer(
            llm=self.llm,
            question=question,
            documents=retrieval.documents,
            chat_history=chat_history,
            system_prompt=self._system_prompt(retrieval),
        )
        return answer, retrieval.documents

    def stream_query(
        self,
        question: str,
        chat_history: str = "",
    ) -> tuple[Iterator[str], list[Document]]:
        """Retrieve now, and return an iterator that streams the answer."""
        retrieval = self._retrieve(question, chat_history)
        stream = stream_answer(
            llm=self.llm,
            question=question,
            documents=retrieval.documents,
            chat_history=chat_history,
            system_prompt=self._system_prompt(retrieval),
        )
        return stream, retrieval.documents

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
