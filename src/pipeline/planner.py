"""
Query planner — query translation, logical routing and self-query in ONE LLM call

Each of the three stages needs a small LLM decision about the same
question. Asking for all three in one structured reply removes two
round-trips (and two requests against the rate limit) from every question,
while the stages stay separate in the code and in the streamed trace:

* ``queries``     — query translation, worded by the configured strategy
* ``datasources`` — logical routing over the stores that hold data
* ``filter``      — self-query metadata filter for the vectorstore

When nothing needs deciding (simple strategy, one store, nothing to filter)
no call is made at all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.query_construction.self_query import MetadataCatalog, MetadataFilter, filter_from_json
from src.query_translation.base import QueryTranslator, dedupe_queries
from src.routing.logical import VECTORSTORE


# How each translation strategy asks for its queries inside the plan.
TRANSLATION_INSTRUCTIONS = {
    "simple": "Return an empty list.",
    "multi_query": "Write {n} differently worded versions of the question that would retrieve "
                   "relevant passages through similarity search.",
    "rag_fusion": "Write {n} diverse search queries that together cover the question from "
                  "different angles.",
    "decomposition": "Break the question into up to {n} sub-questions that can each be answered "
                     "on their own. If it is already simple, return it unchanged.",
    "step_back": "Write ONE more generic step-back question about the background the question "
                 "depends on.",
    "hyde": "Write ONE short hypothetical passage (3-5 sentences) that would answer the question, "
            "as it might appear in a document.",
}

_PLAN_PROMPT = ChatPromptTemplate.from_template("""\
You plan the search for a question-answering system. Return ONLY a JSON \
object with three keys.

"queries": a list of strings. {translation}

"datasources": a list choosing where to search, most relevant first, from:
{sources}
Choose relational_db for figures, totals, rankings or comparisons the tables \
can answer; graph_db for how entities are related or connected; the \
vectorstore for everything else, including explanations. Several may be chosen.

"filter": {filter_help}

Question: {question}""")

_FILTER_HELP = """\
an object restricting the vectorstore search, ONLY when the question \
explicitly asks for it (e.g. "in the pricing PDF", "from the tables", "on \
page 4"); otherwise {{}}. Optional keys: "source" (list of file names from: \
{sources}), "type" (list from: {types}), "page" (list of integers 1-{max_page})."""


@dataclass
class Plan:
    queries: list[str]
    datasources: list[str]
    metadata_filter: MetadataFilter = field(default_factory=MetadataFilter)
    planned: bool = True        # False when no LLM call was needed or it failed


class QueryPlanner:
    def __init__(self, llm: BaseChatModel, strategy: str, num_queries: int = 4):
        self._chain = _PLAN_PROMPT | llm | StrOutputParser()
        self._strategy = strategy
        self._num_queries = num_queries

    def needs_llm(self, available: dict[str, str], catalog: MetadataCatalog, use_filter: bool) -> bool:
        return self._strategy != "simple" or len(available) > 1 or (use_filter and catalog.filterable)

    def plan(
        self,
        question: str,
        translator: QueryTranslator,
        available: dict[str, str],
        catalog: MetadataCatalog,
        use_filter: bool = True,
    ) -> Plan:
        # Without a plan, search the vectorstore: querying every store would
        # spend extra LLM calls (Text-to-SQL/Cypher) on a guess.
        fallback = Plan([question], [VECTORSTORE] if VECTORSTORE in available else list(available)[:1],
                        planned=False)
        if not self.needs_llm(available, catalog, use_filter):
            return fallback

        if use_filter and catalog.filterable:
            filter_help = _FILTER_HELP.format(
                sources=", ".join(Path(s).name for s in catalog.sources),
                types=", ".join(catalog.types),
                max_page=max(catalog.max_page, 1),
            )
        else:
            filter_help = "always {}."
        try:
            reply = self._chain.invoke({
                "question": question,
                "translation": TRANSLATION_INSTRUCTIONS[self._strategy].format(n=self._num_queries),
                "sources": "\n".join(f"- {name}: {desc}" for name, desc in available.items()),
                "filter_help": filter_help,
            })
            data = _parse_json(reply)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN]  Query planning failed ({e}); searching the vectorstore with the question.")
            return fallback

        extra = [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()]
        queries = translator.from_plan(question, extra[: self._num_queries])

        chosen = [str(d).strip().lower() for d in (data.get("datasources") or [])]
        datasources = [d for d in dict.fromkeys(chosen) if d in available]
        if not datasources:
            datasources = [VECTORSTORE] if VECTORSTORE in available else list(available)

        metadata_filter = (
            filter_from_json(data.get("filter") or {}, catalog)
            if use_filter and catalog.filterable else MetadataFilter()
        )
        return Plan(dedupe_queries(queries), datasources, metadata_filter)


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object in the reply")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("the reply is not a JSON object")
    return data
