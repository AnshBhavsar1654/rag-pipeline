"""
RAG Pipeline — LLM Generation

Configurable LLM generation with support for OpenAI, Groq, Google and Ollama providers.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import LLMConfig


def create_llm(config: LLMConfig) -> BaseChatModel:
    """Create an LLM instance based on configuration.

    With ``config.fallback`` set, returns a runnable that retries a failed
    call on the fallback model (still usable anywhere a chat model is).
    """

    if config.provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            max_retries=config.max_retries,
            **({"base_url": config.base_url} if config.base_url else {}),
        )
        print(f"[LLM] LLM: OpenAI ({config.model})")

    elif config.provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            max_retries=config.max_retries,
        )
        print(f"[LLM] LLM: Groq ({config.model})")

    elif config.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=config.model,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            max_retries=config.max_retries,
        )
        print(f"[LLM] LLM: Google ({config.model})")

    elif config.provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama
        llm = ChatOllama(
            model=config.model,
            temperature=config.temperature,
            base_url=config.base_url or "http://localhost:11434",
        )
        print(f"[LLM] LLM: Ollama ({config.model})")

    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")

    if config.fallback is not None:
        return llm.with_fallbacks([create_llm(config.fallback)])
    return llm



# ---------------------------------------------------------------------------
# Generation chain
# ---------------------------------------------------------------------------

_GROUNDING_RULES = """\
Answer using ONLY the numbered context passages provided by the user. Cite \
the passages that support each claim with their numbers in square brackets, \
e.g. [1] or [2][3], using plain ASCII square brackets. If the passages do not \
contain the answer, say that you could not find it in the documents rather \
than guessing. Never invent \
sources or facts."""

# Every piece of user- or config-supplied text enters the prompt as a
# variable, never by string formatting, so braces in it can't break the template.
_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}\n\n{grounding_rules}"),
    ("human", "Context passages:\n\n{context}\n\n{history_block}Question: {question}"),
])

SOURCES_HEADER = "**Sources**"


def source_label(doc: Document) -> str:
    """Short human-readable origin of a chunk, e.g. ``report.pdf, p. 3 (table)``."""
    meta = doc.metadata
    source = str(meta.get("source", "Unknown"))
    label = source if source.startswith(("http://", "https://")) else Path(source).name
    page = meta.get("page")
    if isinstance(page, int) and page > 0:
        label += f", p. {page}"
    kind = meta.get("type")
    if kind and kind != "text":
        label += f" ({kind.replace('_', ' ')})"
    return label


def format_context(documents: list[Document]) -> str:
    """Number each chunk so the model can cite it and the UI can list it."""
    if not documents:
        return "(No relevant passages were found.)"
    return "\n\n".join(
        f"[{i}] Source: {source_label(doc)}\n{doc.page_content}"
        for i, doc in enumerate(documents, start=1)
    )


def _prompt_inputs(
    question: str,
    documents: list[Document],
    chat_history: str,
    system_prompt: str,
) -> dict[str, str]:
    history_block = (
        f"Conversation so far:\n{chat_history}\n\n" if chat_history.strip() else ""
    )
    return {
        "system_prompt": system_prompt.strip() or "You are a helpful AI assistant.",
        "grounding_rules": _GROUNDING_RULES,
        "context": format_context(documents),
        "history_block": history_block,
        "question": question,
    }


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
    chain = _RAG_PROMPT | llm | StrOutputParser()
    return chain.invoke(_prompt_inputs(question, documents, chat_history, system_prompt))


def stream_answer(
    llm: BaseChatModel,
    question: str,
    documents: list[Document],
    chat_history: str = "",
    system_prompt: str = "",
) -> Iterator[str]:
    """Like :func:`generate_answer`, but yields the answer as it is generated."""
    chain = _RAG_PROMPT | llm | StrOutputParser()
    yield from chain.stream(_prompt_inputs(question, documents, chat_history, system_prompt))


def format_sources(documents: list[Document], snippet_chars: int = 150) -> str:
    """List the context passages, numbered to match the answer's [n] citations."""
    lines: list[str] = []
    for i, doc in enumerate(documents, start=1):
        snippet = " ".join(doc.page_content.split())[:snippet_chars]
        lines.append(f"- **[{i}]** {source_label(doc)} — \"{snippet}...\"")
    return "\n".join(lines)
