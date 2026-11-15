"""
RAG Pipeline — Main Orchestrator

Ties together all components: document loading, chunking, embedding,
vector store, retrieval strategies, re-ranking, and generation.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from src.config import RAGConfig, load_config
from src.indexing.loaders.loader import load_documents, _load_from_file
from src.indexing.chunking import split_documents
from src.indexing.embeddings import create_embeddings, max_input_tokens
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
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.ranking import Reranker
from src.query_translation import create_translator
from src.query_translation.base import QueryTranslator
from src.query_translation.condense import QuestionCondenser
from src.generation.generator import (
    SOURCES_HEADER,
    create_llm,
    format_sources,
    generate_answer,
    stream_answer,
)


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

        # 2. Split into chunks
        chunks = self._split(documents)
        if not chunks:
            print("\n[ERROR] No chunks generated. Check chunking settings.")
            return

        # 3. Embed and store
        if rebuild:
            self._vectorstore = build_vectorstore(
                self.config.vectorstore,
                self.embeddings,
                self.config.embedding.signature,
                documents=chunks,
            )
        else:
            self._open_for_writing()
            delete_sources(self._vectorstore, {str(c.metadata.get("source", "")) for c in chunks})
            written = add_documents(self._vectorstore, chunks)
            total = self._vectorstore._collection.count()
            print(f"[OK] Indexed {written} chunks ({total} in the vector store)")
        self._init_retrieval()

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
            documents=self._split(old_chunks),
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

        chunks = self._split(all_docs)

        self._open_for_writing()
        written = add_documents(self._vectorstore, chunks)

        # Rebuild retrieval (including the keyword index) so new docs are searchable
        self._init_retrieval()

        msg = f"Successfully ingested {len(loaded_names)} file(s) ({written} chunks): {', '.join(loaded_names)}"
        if failed_names:
            msg += f"\nFailed: {', '.join(failed_names)}"
        return msg

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
        return split_documents(documents, chunking)

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
        self._base_retriever = HybridRetriever(
            self._vectorstore,
            documents=all_documents(self._vectorstore) if hybrid.enabled else [],
            k=per_query_k,
            fetch_k=hybrid.fetch_k,
            rrf_k=hybrid.rrf_k,
            use_bm25=hybrid.enabled,
        )
        self._translator = create_translator(retrieval, self.llm)

    # -----------------------------------------------------------------
    # Query
    # -----------------------------------------------------------------

    def retrieve(self, question: str, chat_history: str = "") -> list[Document]:
        """Find the chunks to answer *question* with: rewrite → search → rerank → cap."""
        if self._base_retriever is None:
            raise RuntimeError("Pipeline not loaded. Call .load() or .ingest() first.")

        # 1. Make follow-ups standalone so the search sees the full intent
        search_query = question
        if chat_history and self.config.chat.rewrite_followups:
            if self._condenser is None:
                self._condenser = QuestionCondenser(self.llm)
            search_query = self._condenser.condense(question, chat_history)
            if search_query != question:
                print(f"[QUERY] Rewrote follow-up as: {search_query}")

        # 2. Query translation, then search with every query and fuse
        queries = self._translator.translate(search_query)
        documents = self._translator.fuse([self._base_retriever.invoke(q) for q in queries])

        # 3. Re-rank a candidate pool, or just keep the best top_k
        if self._reranker is not None and self._reranker.enabled:
            candidates = documents[: self.config.reranking.candidates]
            return self._reranker.rerank(search_query, candidates)
        return documents[: self.config.retrieval.top_k]

    def query(
        self,
        question: str,
        chat_history: str = "",
    ) -> tuple[str, list[Document]]:
        """Run the full RAG query pipeline.

        Args:
            question: The user's question.
            chat_history: Optional formatted conversation history.

        Returns:
            (answer, source_documents)
        """
        documents = self.retrieve(question, chat_history)
        answer = generate_answer(
            llm=self.llm,
            question=question,
            documents=documents,
            chat_history=chat_history,
            system_prompt=self.config.chat.system_prompt,
        )
        return answer, documents

    def stream_query(
        self,
        question: str,
        chat_history: str = "",
    ) -> tuple[Iterator[str], list[Document]]:
        """Retrieve now, and return an iterator that streams the answer."""
        documents = self.retrieve(question, chat_history)
        stream = stream_answer(
            llm=self.llm,
            question=question,
            documents=documents,
            chat_history=chat_history,
            system_prompt=self.config.chat.system_prompt,
        )
        return stream, documents

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
