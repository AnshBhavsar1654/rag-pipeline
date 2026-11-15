"""
Query Construction — Text-to-SQL

Turns a natural-language question into a SQLite SELECT over the tables in
the relational DB, runs it read-only, and returns the result as a document
the generator can cite. A failing query gets one retry with the error.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.indexing.loaders.table_extractor import table_to_markdown
from src.stores.relational import RelationalStore


_SQL_PROMPT = ChatPromptTemplate.from_template("""\
You write SQLite queries. Given the database schema below, write ONE SELECT \
query that answers the question. Use only the tables and columns shown, and \
quote identifiers with double quotes. Text comparisons should be \
case-insensitive (use LIKE or lower()). Output ONLY the SQL, no explanation.

Schema:
{schema}

Question: {question}
{error_block}SQL:""")


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply that may wrap it in a code fence."""
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    sql = fenced.group(1) if fenced else text
    sql = sql.strip()
    match = re.search(r"(?is)\b(with|select)\b.*", sql)
    return match.group(0).strip().rstrip(";") if match else sql


class TextToSQL:
    def __init__(self, llm: BaseChatModel, store: RelationalStore, max_rows: int = 50):
        self._chain = _SQL_PROMPT | llm | StrOutputParser()
        self._store = store
        self._max_rows = max_rows

    def run(self, question: str) -> tuple[str, list[Document]]:
        """Return (sql, documents); documents is empty if no query worked."""
        schema = self._store.schema()
        if not schema:
            return "", []
        error_block, sql = "", ""
        for _ in range(2):
            try:
                sql = extract_sql(self._chain.invoke({
                    "schema": schema, "question": question, "error_block": error_block,
                }))
            except Exception as e:  # noqa: BLE001 - the LLM call itself failed: retrying won't help
                print(f"[WARN]  Text-to-SQL could not reach the LLM: {e}")
                return "", []
            try:
                columns, rows = self._store.run_select(sql, self._max_rows)
            except Exception as e:
                print(f"[WARN]  Text-to-SQL attempt failed: {e}")
                error_block = f"\nYour previous query:\n{sql}\nfailed with: {e}\nFix it.\n"
                continue
            return sql, [self._to_document(sql, columns, rows)]
        return sql, []

    def _to_document(self, sql: str, columns: list[str], rows: list[tuple]) -> Document:
        if rows:
            body = table_to_markdown([columns, *[[str(v) for v in r] for r in rows]])
        else:
            body = "(the query returned no rows)"
        tables = self._store.sources_for(sql)
        return Document(
            page_content=f"Result of the SQL query `{sql}` on the relational DB:\n\n{body}",
            metadata={
                "source": "Relational DB" + (f" ({', '.join(tables)})" if tables else ""),
                "type": "sql_result",
                "sql": sql,
            },
        )
