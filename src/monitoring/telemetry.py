"""
Monitoring — telemetry store

Every answered question and every indexed upload is recorded in a local
SQLite file, tagged with the git commit the code ran at. That turns "it
feels faster" into numbers you can compare between versions:

    python -m src.monitoring.report

Nothing leaves the machine. Question text is stored only when
``monitoring.store_questions`` is on.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def code_version() -> str:
    """Short git commit of the running code, with "+dirty" for uncommitted edits."""
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return (commit or "unknown") + ("+dirty" if dirty else "")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY, ts REAL, version TEXT, question TEXT, strategy TEXT,
    datasources TEXT, total_ms INTEGER, first_token_ms INTEGER, stage_ms TEXT,
    llm_calls INTEGER, failovers INTEGER, input_tokens INTEGER, output_tokens INTEGER,
    documents INTEGER, context_chars INTEGER, answer_chars INTEGER, self_rag_retries INTEGER,
    model TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS ingestions (
    id INTEGER PRIMARY KEY, ts REAL, version TEXT, files TEXT, bytes INTEGER, pages INTEGER,
    chars INTEGER, chunks INTEGER, searchable_s REAL, enrich_s REAL, stage_s TEXT,
    llm_calls INTEGER, input_tokens INTEGER, output_tokens INTEGER, error TEXT
);
"""


class Telemetry:
    def __init__(self, path: str | Path, enabled: bool = True, store_questions: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self.store_questions = store_questions
        self._lock = threading.Lock()

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        try:
            conn.executescript(_SCHEMA)
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _insert(self, table: str, row: dict) -> None:
        if not self.enabled:
            return
        row = {"ts": time.time(), "version": code_version(), **row}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value)
        cols = ", ".join(row)
        marks = ", ".join("?" * len(row))
        try:
            with self._lock, self._connect() as conn:
                conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))
        except sqlite3.Error as e:  # monitoring must never break answering
            print(f"[WARN]  Could not record telemetry: {e}")

    def record_query(self, **row) -> None:
        if not self.store_questions:
            row["question"] = None
        self._insert("queries", row)

    def record_ingestion(self, **row) -> None:
        self._insert("ingestions", row)

    def rows(self, table: str, limit: int | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            sql = f"SELECT * FROM {table} ORDER BY ts DESC" + (f" LIMIT {int(limit)}" if limit else "")
            return [dict(r) for r in conn.execute(sql)]


def percentile(values: list[float], q: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summarize(telemetry: Telemetry) -> dict:
    """Per-version summary of recorded questions and uploads."""
    out: dict[str, dict] = {}
    for row in telemetry.rows("queries"):
        out.setdefault(row["version"], {"queries": [], "ingestions": []})["queries"].append(row)
    for row in telemetry.rows("ingestions"):
        out.setdefault(row["version"], {"queries": [], "ingestions": []})["ingestions"].append(row)

    summary = {}
    for version, data in out.items():
        q, ing = data["queries"], data["ingestions"]
        ok = [r for r in q if not r["error"]]
        n = len(q)
        pages = [r["searchable_s"] / r["pages"] for r in ing if r["pages"] and r["searchable_s"] is not None]
        summary[version] = {
            "first_seen": min((r["ts"] for r in q + ing), default=None),
            "questions": n,
            "error_rate": round(1 - len(ok) / n, 3) if n else None,
            "total_ms_p50": percentile([r["total_ms"] for r in ok], 0.5),
            "total_ms_p95": percentile([r["total_ms"] for r in ok], 0.95),
            "first_token_ms_p50": percentile([r["first_token_ms"] for r in ok], 0.5),
            "llm_calls_avg": round(sum(r["llm_calls"] or 0 for r in q) / n, 2) if n else None,
            "tokens_avg": round(sum((r["input_tokens"] or 0) + (r["output_tokens"] or 0) for r in q) / n)
            if n else None,
            "failover_rate": round(sum(1 for r in q if r["failovers"]) / n, 3) if n else None,
            "uploads": len(ing),
            "searchable_s_per_page_p50": percentile(pages, 0.5),
            "enrich_llm_calls_avg": round(sum(r["llm_calls"] or 0 for r in ing) / len(ing), 1) if ing else None,
        }
    return dict(sorted(summary.items(), key=lambda kv: kv[1]["first_seen"] or 0))
