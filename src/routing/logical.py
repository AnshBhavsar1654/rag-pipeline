"""
Routing — Logical routing

Lets the LLM choose which data store(s) should answer the question: the
vectorstore (text passages), the relational DB (tables, via Text-to-SQL)
or the graph DB (entities and relationships, via Text-to-Cypher).
"""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


VECTORSTORE = "vectorstore"
RELATIONAL_DB = "relational_db"
GRAPH_DB = "graph_db"

_ROUTE_PROMPT = ChatPromptTemplate.from_template("""\
You are an expert at routing a user question to the right data source(s).

Data sources:
{sources}

Pick the source(s) most likely to contain the answer. Choose the \
relational_db for questions about figures, totals, rankings or comparisons \
that the tables can answer; the graph_db for questions about how entities \
are related or connected; the vectorstore for everything else, including \
explanations. Several sources may be chosen when the question needs them.

Reply with the chosen source names only, comma separated, most relevant first.

Question: {question}""")


class LogicalRouter:
    def __init__(self, llm: BaseChatModel):
        self._chain = _ROUTE_PROMPT | llm | StrOutputParser()

    def route(self, question: str, available: dict[str, str]) -> list[str]:
        """Return the chosen data sources, a subset of *available* (name → description)."""
        if len(available) <= 1:
            return list(available)
        sources = "\n".join(f"- {name}: {desc}" for name, desc in available.items())
        try:
            reply = self._chain.invoke({"question": question, "sources": sources})
        except Exception as e:
            print(f"[WARN]  Logical routing failed ({e}); using every data source.")
            return list(available)
        found = sorted(
            (m.start(), name)
            for name in available
            if (m := re.search(rf"\b{name}\b", reply.lower()))
        )
        chosen = [name for _, name in found]
        return chosen or [VECTORSTORE if VECTORSTORE in available else next(iter(available))]
