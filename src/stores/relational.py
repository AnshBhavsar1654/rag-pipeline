"""
Relational DB — SQLite store for tabular data

Every table found while indexing (Markdown tables pulled out of PDFs and
DOCX files, and CSV files) is loaded into its own SQLite table so that
questions about numbers, totals and comparisons can be answered with SQL
instead of fuzzy text search. A ``_catalog`` table remembers which file
(and page) each table came from.
"""

from __future__ import annotations

import csv
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document


_CATALOG = "_catalog"


@dataclass
class TableData:
    source: str
    columns: list[str]
    rows: list[list[str]]
    page: int | None = None
    name_hint: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Finding tables in loaded documents
# ---------------------------------------------------------------------------

def parse_markdown_table(text: str) -> tuple[list[str], list[list[str]]] | None:
    """Parse a pipe-delimited Markdown table into (header, rows)."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return None

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header = cells(lines[0])
    body = [ln for ln in lines[1:] if not re.fullmatch(r"\|?[\s:\-|]+\|?", ln)]
    rows = [cells(ln) for ln in body]
    return (header, rows) if rows else None


def extract_tables(documents: list[Document]) -> list[TableData]:
    """Collect the tables in *documents*: Markdown table chunks and CSV files."""
    tables: list[TableData] = []
    seen_csv: set[str] = set()
    for doc in documents:
        meta = doc.metadata
        source = str(meta.get("source", ""))
        if meta.get("type") == "table":
            parsed = parse_markdown_table(doc.page_content)
            if parsed:
                page = meta.get("page") if isinstance(meta.get("page"), int) else None
                index = meta.get("table_index", len(tables))
                tables.append(TableData(
                    source=source, columns=parsed[0], rows=parsed[1], page=page,
                    name_hint=f"{Path(source).stem}_{'p' + str(page) + '_' if page else ''}t{index}",
                ))
        elif source.lower().endswith(".csv") and source not in seen_csv:
            seen_csv.add(source)
            table = _read_csv(source)
            if table:
                tables.append(table)
    return tables


def _read_csv(path: str) -> TableData | None:
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = list(csv.reader(fh))
    except OSError as e:
        print(f"  [WARN]  Could not read CSV {path} for the relational DB: {e}")
        return None
    if len(reader) < 2:
        return None
    return TableData(source=path, columns=reader[0], rows=reader[1:], name_hint=Path(path).stem)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def _identifier(text: str, fallback: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    if not name:
        name = fallback
    if name[0].isdigit():
        name = f"t_{name}"
    return name


def _value(cell: str):
    """Store numbers as numbers so SQL can sum and compare them."""
    text = cell.strip()
    cleaned = re.sub(r"[,$€£₹%\s]", "", text)
    if re.fullmatch(r"-?\d+", cleaned):
        return int(cleaned)
    if re.fullmatch(r"-?\d*\.\d+", cleaned):
        return float(cleaned)
    return text


class RelationalStore:
    """A small SQLite database holding every extracted table."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _connect(self, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            return sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {_CATALOG} "
            "(table_name TEXT PRIMARY KEY, source TEXT, page INTEGER, columns TEXT)"
        )
        return conn

    # -- writing ------------------------------------------------------------

    def add_tables(self, tables: list[TableData]) -> list[str]:
        """Create one SQL table per extracted table; returns the table names."""
        if not tables:
            return []
        names: list[str] = []
        with self._connect() as conn:
            existing = {r[0] for r in conn.execute(f"SELECT table_name FROM {_CATALOG}")}
            for table in tables:
                base = _identifier(table.name_hint or Path(table.source).stem, "table")
                name, n = base, 2
                while name in existing:
                    name, n = f"{base}_{n}", n + 1
                existing.add(name)

                columns: list[str] = []
                for i, col in enumerate(table.columns):
                    col_name = _identifier(col, f"col_{i + 1}")
                    while col_name in columns:
                        col_name += f"_{i + 1}"
                    columns.append(col_name)

                width = len(columns)
                rows = [
                    [_value(c) for c in (row + [""] * width)[:width]]
                    for row in table.rows
                    if any(c.strip() for c in row)
                ]
                cols_sql = ", ".join(f'"{c}"' for c in columns)
                conn.execute(f'CREATE TABLE "{name}" ({cols_sql})')
                conn.executemany(
                    f'INSERT INTO "{name}" VALUES ({", ".join("?" * width)})', rows,
                )
                conn.execute(
                    f"INSERT INTO {_CATALOG} VALUES (?, ?, ?, ?)",
                    (name, table.source, table.page, ", ".join(table.columns)),
                )
                names.append(name)
        return names

    def delete_sources(self, sources: set[str]) -> None:
        """Drop every table that came from one of *sources* (before re-adding)."""
        if not sources or not self.path.exists():
            return
        with self._connect() as conn:
            marks = ", ".join("?" * len(sources))
            stale = [r[0] for r in conn.execute(
                f"SELECT table_name FROM {_CATALOG} WHERE source IN ({marks})", sorted(sources),
            )]
            for name in stale:
                conn.execute(f'DROP TABLE IF EXISTS "{name}"')
            conn.execute(f"DELETE FROM {_CATALOG} WHERE source IN ({marks})", sorted(sources))

    def reset(self) -> None:
        if self.path.exists():
            self.path.unlink()

    # -- reading ------------------------------------------------------------

    def table_names(self) -> list[str]:
        if not self.path.exists():
            return []
        with self._connect(readonly=True) as conn:
            return [r[0] for r in conn.execute(f"SELECT table_name FROM {_CATALOG} ORDER BY table_name")]

    @property
    def is_empty(self) -> bool:
        return not self.table_names()

    def schema(self, sample_rows: int = 3) -> str:
        """CREATE statements plus a few sample rows, the context Text-to-SQL needs."""
        if self.is_empty:
            return ""
        parts: list[str] = []
        with self._connect(readonly=True) as conn:
            catalog = conn.execute(
                f"SELECT table_name, source, page FROM {_CATALOG} ORDER BY table_name"
            ).fetchall()
            for name, source, page in catalog:
                cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{name}")')]
                origin = Path(source).name + (f", page {page}" if page else "")
                parts.append(f'-- from {origin}\nCREATE TABLE "{name}" ({", ".join(cols)});')
                samples = conn.execute(f'SELECT * FROM "{name}" LIMIT {sample_rows}').fetchall()
                if samples:
                    parts.append("-- sample rows:\n" + "\n".join(f"-- {tuple(r)}" for r in samples))
        return "\n".join(parts)

    def run_select(self, sql: str, max_rows: int = 50) -> tuple[list[str], list[tuple]]:
        """Run one read-only SELECT and return (column names, rows)."""
        statement = sql.strip().rstrip(";").strip()
        if ";" in statement or not re.match(r"(?is)^\s*(select|with)\b", statement):
            raise ValueError("Only a single SELECT statement is allowed.")
        with self._connect(readonly=True) as conn:
            cursor = conn.execute(statement)
            columns = [d[0] for d in cursor.description or []]
            return columns, cursor.fetchmany(max_rows)

    def sources_for(self, sql: str) -> list[str]:
        """Original files of the tables a query mentions (for citations)."""
        with self._connect(readonly=True) as conn:
            catalog = conn.execute(f"SELECT table_name, source FROM {_CATALOG}").fetchall()
        lowered = sql.lower()
        return sorted({Path(src).name for name, src in catalog if name.lower() in lowered})
