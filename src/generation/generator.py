"""
RAG Pipeline — LLM Generation

Configurable LLM generation with support for OpenAI, Groq, Google and Ollama providers.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import LLMConfig


def create_llm(config: LLMConfig) -> BaseChatModel:
    """Create an LLM instance based on configuration."""

    if config.provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        print(f"[LLM] LLM: OpenAI ({config.model})")

    elif config.provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        print(f"[LLM] LLM: Groq ({config.model})")

    elif config.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=config.model,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )
        print(f"[LLM] LLM: Google ({config.model})")

    elif config.provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        llm = ChatOllama(
            model=config.model,
            temperature=config.temperature,
            base_url=config.base_url or "http://localhost:11434",
        )
        print(f"[LLM] LLM: Ollama ({config.model})")

    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")

    return llm


# ---------------------------------------------------------------------------
# Generation chain
# ---------------------------------------------------------------------------

_RAG_TEMPLATE = """\
You are a helpful AI assistant. Answer the question based ONLY on the \
following context. If the context doesn't contain enough information to \
answer the question, say so honestly.

Context:
{context}

Question: {question}

Answer:"""


_RAG_TEMPLATE_WITH_HISTORY = """\
You are a helpful AI assistant. Answer the question based on the provided \
context and conversation history. If the context doesn't contain enough \
information, say so honestly.

Context:
{context}

Conversation History:
{chat_history}

Question: {question}

Answer:"""


def generate_answer(
    llm: BaseChatModel,
    question: str,
    documents: list[Document],
    chat_history: str = "",
    system_prompt: str = "",
) -> str:
    """Generate an answer using the LLM given retrieved documents.

    Args:
        llm: The language model to use.
        question: The user's question.
        documents: Retrieved context documents.
        chat_history: Optional formatted chat history string.
        system_prompt: Optional system prompt override.

    Returns:
        The generated answer string.
    """
    context = "\n\n---\n\n".join(doc.page_content for doc in documents)

    if chat_history:
        template = _RAG_TEMPLATE_WITH_HISTORY
        invoke_args = {
            "context": context,
            "chat_history": chat_history,
            "question": question,
        }
    else:
        template = _RAG_TEMPLATE
        invoke_args = {
            "context": context,
            "question": question,
        }

    # Override template if a custom system prompt is provided
    if system_prompt:
        if chat_history:
            template = (
                f"{system_prompt}\n\n"
                "Context:\n{context}\n\n"
                "Conversation History:\n{chat_history}\n\n"
                "Question: {question}\n\nAnswer:"
            )
        else:
            template = (
                f"{system_prompt}\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\nAnswer:"
            )

    chain = (
        ChatPromptTemplate.from_template(template)
        | llm
        | StrOutputParser()
    )

    return chain.invoke(invoke_args)


def format_sources(documents: list[Document], max_sources: int = 3) -> str:
    """Format source documents for display."""
    if not documents:
        return ""

    sources: list[str] = []
    seen: set[str] = set()

    for doc in documents[:max_sources]:
        # Build source identifier from metadata
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")

        identifier = f"{source}"
        if page:
            identifier += f" (page {page})"

        if identifier not in seen:
            seen.add(identifier)
            snippet = doc.page_content[:150].replace("\n", " ").strip()
            sources.append(f"  [SRC] {identifier}\n     \"{snippet}...\"")

    return "\n".join(sources)
