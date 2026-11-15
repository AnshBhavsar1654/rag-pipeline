"""Offline tests for the relational DB and Text-to-SQL."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel

from src.query_construction.text_to_sql import TextToSQL, extract_sql
from src.stores.relational import RelationalStore, extract_tables, parse_markdown_table


TABLE_MD = """\
| Model | Params (B) | Vendor |
| --- | --- | --- |
| Alpha | 7 | Acme |
| Beta | 70 | Acme |
| Gamma | 13 | Zeta |"""


@pytest.fixture
def store(tmp_path):
    store = RelationalStore(tmp_path / "rel.sqlite")
    doc = Document(page_content=TABLE_MD, metadata={"source": "specs.pdf", "type": "table", "page": 2})
    store.add_tables(extract_tables([doc]))
    return store


def test_markdown_table_round_trips():
    header, rows = parse_markdown_table(TABLE_MD)
    assert header == ["Model", "Params (B)", "Vendor"]
    assert rows[1] == ["Beta", "70", "Acme"]


def test_tables_are_queryable_with_numbers_as_numbers(store):
    (name,) = store.table_names()
    cols, rows = store.run_select(f'SELECT SUM(params_b) FROM "{name}" WHERE vendor = \'Acme\'')
    assert rows == [(77,)]
    assert "specs.pdf, page 2" in store.schema()


def test_only_single_selects_are_allowed(store):
    with pytest.raises(ValueError):
        store.run_select("DROP TABLE _catalog")
    with pytest.raises(ValueError):
        store.run_select("SELECT 1; DELETE FROM _catalog")


def test_delete_sources_drops_their_tables(store):
    store.delete_sources({"specs.pdf"})
    assert store.is_empty


def test_csv_files_become_tables(tmp_path):
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text("region,revenue\nnorth,100\nsouth,250\n", encoding="utf-8")
    store = RelationalStore(tmp_path / "rel.sqlite")
    store.add_tables(extract_tables([Document(page_content="x", metadata={"source": str(csv_path)})]))
    assert store.run_select("SELECT MAX(revenue) FROM sales")[1] == [(250,)]


def test_text_to_sql_retries_after_an_error(store):
    (name,) = store.table_names()
    llm = FakeListChatModel(responses=[
        "SELECT nope FROM missing",
        f'```sql\nSELECT model FROM "{name}" ORDER BY params_b DESC LIMIT 1;\n```',
    ])
    sql, docs = TextToSQL(llm, store).run("Which model is largest?")
    assert "ORDER BY" in sql
    assert "Beta" in docs[0].page_content
    assert docs[0].metadata["type"] == "sql_result"


def test_extract_sql_strips_chatter():
    assert extract_sql("Sure! Here it is:\nSELECT 1;") == "SELECT 1"
