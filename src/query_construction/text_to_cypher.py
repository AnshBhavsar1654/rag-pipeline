"""
Query Construction — Text-to-Cypher

Turns a natural-language question into a Cypher query over the knowledge
graph. If the query fails or finds nothing, falls back to the facts around
every entity the question mentions, so the graph still contributes.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.stores.graph import GraphStore


_CYPHER_PROMPT = ChatPromptTemplate.from_template("""\
You write Cypher queries for a knowledge graph. Every node has a lower-case \
`name` property and a label; relationships have a type. Write ONE read-only \
query (MATCH ... WHERE ... RETURN ... LIMIT ...) that answers the question.
Rules:
- Compare names with lower-case strings, e.g. WHERE a.name == "gpt-4", or \
use CONTAINS for partial matches.
- Use only the labels and relationship types in the schema.
- Use exactly ONE MATCH clause containing ONE path pattern, e.g. MATCH (a)-[r:TYPE]->(b) or MATCH (a)-[r1]->(b)<-[r2]-(c). No commas between patterns, no second MATCH, no OPTIONAL MATCH, no WITH.
- RETURN node names (e.g. a.name, b.name), not whole nodes.
- Output ONLY the Cypher query.

Schema:
{schema}

Question: {question}
{error_block}Cypher:""")


def extract_cypher(text: str) -> str:
    fenced = re.search(r"```(?:cypher)?\s*(.*?)```", text, re.S | re.I)
    query = (fenced.group(1) if fenced else text).strip()
    match = re.search(r"(?is)\bmatch\b.*", query)
    return (match.group(0) if match else query).strip().rstrip(";")


class TextToCypher:
    def __init__(self, llm: BaseChatModel, store: GraphStore, max_results: int = 50):
        self._chain = _CYPHER_PROMPT | llm | StrOutputParser()
        self._store = store
        self._max_results = max_results

    def run(self, question: str) -> tuple[str, list[Document]]:
        """Return (cypher, documents); the cypher is empty when the fallback was used."""
        schema = self._store.schema()
        if not schema:
            return "", []
        error_block, cypher = "", ""
        for _ in range(2):
            try:
                cypher = extract_cypher(self._chain.invoke({
                    "schema": schema, "question": question, "error_block": error_block,
                }))
                if re.search(r"(?i)\b(create|merge|delete|set|remove|drop)\b", cypher):
                    raise ValueError("Only read-only MATCH queries are allowed.")
                rows = self._store.run_cypher(cypher)[: self._max_results]
            except Exception as e:
                print(f"[WARN]  Text-to-Cypher attempt failed: {e}")
                error_block = f"\nYour previous query:\n{cypher}\nfailed with: {e}\nFix it.\n"
                continue
            if rows:
                lines = [" | ".join(f"{k}: {v}" for k, v in row.items()) for row in rows]
                return cypher, [self._document(
                    f"Result of the Cypher query `{cypher}` on the graph DB:\n" + "\n".join(lines),
                    cypher,
                )]
            break

        facts = self._store.neighborhood(question, limit=self._max_results)
        if not facts:
            return cypher, []
        return "", [self._document(
            "Facts from the knowledge graph about entities in the question:\n" + "\n".join(facts),
            "",
        )]

    @staticmethod
    def _document(text: str, cypher: str) -> Document:
        return Document(
            page_content=text,
            metadata={"source": "Graph DB", "type": "graph_result", "cypher": cypher},
        )
