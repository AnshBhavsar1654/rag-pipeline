"""
RAG Pipeline — Main Orchestrator

Ties together all components: document loading, chunking, embedding,
vector store, retrieval strategies, re-ranking, and generation.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from src.config import RAGConfig, load_config
from src.document_loaders.loader import load_documents, _load_from_file
from src.chunking.splitter import split_documents
from src.embeddings.embedder import create_embeddings
from src.vectorstore.store import create_vectorstore, get_retriever
from src.retrieval.base_retriever import BaseRAGRetriever
from src.retrieval.multi_query import MultiQueryRetriever
from src.retrieval.rag_fusion import RAGFusionRetriever
from src.retrieval.hyde import HyDERetriever
from src.retrieval.reranker import Reranker
from src.generation.generator import create_llm, generate_answer, format_sources


class RAGPipeline:
    """Main RAG pipeline orchestrator.

    Usage:
        # Ingest documents
        pipeline = RAGPipeline.from_config()
        pipeline.ingest()

        # Query
        pipeline = RAGPipeline.from_config()
        pipeline.load()
        answer, sources = pipeline.query("What is X?")
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._llm: BaseChatModel | None = None
        self._retriever: BaseRAGRetriever | None = None
        self._reranker: Reranker | None = None
        self._vectorstore = None
        self._base_retriever = None

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

    # -----------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------

    def ingest(self) -> None:
        """Full ingestion pipeline: load → chunk → embed → store."""
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
        chunks = split_documents(documents, self.config.chunking)
        if not chunks:
            print("\n[ERROR] No chunks generated. Check chunking settings.")
            return

        # 3. Create embeddings
        embeddings = create_embeddings(self.config.embedding)

        # 4. Build and persist vector store
        self._vectorstore = create_vectorstore(
            self.config.vectorstore,
            embeddings,
            documents=chunks,
        )

        print("\n" + "=" * 60)
        print("  [OK] Ingestion complete!")
        print("=" * 60 + "\n")

    def ingest_files(self, file_paths: list[str]) -> str:
        """Ingest uploaded files into the existing vector store.

        Loads, chunks, embeds, and adds the documents to the running
        vector store so the chatbot can answer questions about them.

        Returns a status message describing what happened.
        """
        from pathlib import Path

        if not file_paths:
            return "No files provided."

        # Ensure vector store and embeddings are initialized
        embeddings = create_embeddings(self.config.embedding)
        if self._vectorstore is None:
            self._vectorstore = create_vectorstore(
                self.config.vectorstore, embeddings,
            )

        # Load documents from each file
        all_docs = []
        loaded_names = []
        failed_names = []
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
            return f"Failed to load any documents. Unsupported files: {', '.join(failed_names)}"

        # Split into chunks
        chunks = split_documents(all_docs, self.config.chunking)

        # Add to existing vector store
        self._vectorstore.add_documents(chunks)

        # Refresh retriever so new docs are immediately searchable
        self._base_retriever = get_retriever(
            self._vectorstore,
            top_k=self.config.retrieval.top_k,
        )
        self._retriever = self._create_retriever()

        msg = f"Successfully ingested {len(loaded_names)} file(s) ({len(chunks)} chunks): {', '.join(loaded_names)}"
        if failed_names:
            msg += f"\nFailed: {', '.join(failed_names)}"
        return msg

    # -----------------------------------------------------------------
    # Loading (for query mode)
    # -----------------------------------------------------------------

    def load(self) -> None:
        """Load existing vector store and initialize retrieval components."""
        print("\n[RANK] Loading RAG pipeline...")

        # Embeddings
        embeddings = create_embeddings(self.config.embedding)

        # Vector store
        self._vectorstore = create_vectorstore(
            self.config.vectorstore,
            embeddings,
        )

        # Base LangChain retriever
        self._base_retriever = get_retriever(
            self._vectorstore,
            top_k=self.config.retrieval.top_k,
        )

        # Strategy-specific retriever
        self._retriever = self._create_retriever()

        # Reranker (optional)
        self._reranker = Reranker(self.config.reranking)

        print(f"[STRAT] Retrieval strategy: {self._retriever.strategy_name}")
        print("[OK] Pipeline ready!\n")

    def _create_retriever(self) -> BaseRAGRetriever:
        """Create the appropriate retrieval strategy based on config."""
        strategy = self.config.retrieval.strategy

        if strategy == "simple":
            return _SimpleRetriever(self._base_retriever)

        elif strategy == "multi_query":
            return MultiQueryRetriever(
                retriever=self._base_retriever,
                llm=self.llm,
                num_queries=self.config.retrieval.multi_query.num_queries,
            )

        elif strategy == "rag_fusion":
            return RAGFusionRetriever(
                retriever=self._base_retriever,
                llm=self.llm,
                num_queries=self.config.retrieval.rag_fusion.num_queries,
                rrf_k=self.config.retrieval.rag_fusion.rrf_k,
            )

        elif strategy == "hyde":
            return HyDERetriever(
                retriever=self._base_retriever,
                llm=self.llm,
                hypothesis_prompt=self.config.retrieval.hyde.hypothesis_prompt,
            )

        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

    # -----------------------------------------------------------------
    # Query
    # -----------------------------------------------------------------

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
        if self._retriever is None:
            raise RuntimeError("Pipeline not loaded. Call .load() first.")

        # 1. Retrieve
        documents = self._retriever.retrieve(question)

        # 2. Re-rank (optional)
        if self._reranker and self._reranker._config.enabled:
            documents = self._reranker.rerank(question, documents)

        # 3. Generate answer
        answer = generate_answer(
            llm=self.llm,
            question=question,
            documents=documents,
            chat_history=chat_history,
            system_prompt=self.config.chat.system_prompt,
        )

        return answer, documents

    def format_response(
        self, answer: str, documents: list[Document]
    ) -> str:
        """Format a response with optional source citations."""
        response = answer

        if self.config.chat.show_sources and documents:
            sources = format_sources(documents)
            if sources:
                response += f"\n\n[DOCS] **Sources:**\n{sources}"

        return response


# ---------------------------------------------------------------------------
# Simple retriever (no query transformation)
# ---------------------------------------------------------------------------

class _SimpleRetriever(BaseRAGRetriever):
    """Simple retriever — direct vector similarity search."""

    def __init__(self, retriever):
        self._retriever = retriever

    @property
    def strategy_name(self) -> str:
        return "Simple (Vector Search)"

    def retrieve(self, question: str) -> list[Document]:
        return self._retriever.invoke(question)
