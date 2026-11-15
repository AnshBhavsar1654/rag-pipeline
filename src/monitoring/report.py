"""
Print recorded telemetry per code version.

    python -m src.monitoring.report            # table of every version seen
    python -m src.monitoring.report --json     # the same numbers as JSON
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from src.config import load_config
from src.monitoring.telemetry import Telemetry, summarize

COLUMNS = [
    ("questions", "Questions", "{:.0f}"),
    ("total_ms_p50", "Answer p50 (s)", "ms"),
    ("total_ms_p95", "Answer p95 (s)", "ms"),
    ("first_token_ms_p50", "First words p50 (s)", "ms"),
    ("llm_calls_avg", "LLM calls / question", "{:.2f}"),
    ("tokens_avg", "Tokens / question", "{:.0f}"),
    ("error_rate", "Errors", "pct"),
    ("failover_rate", "Needed failover", "pct"),
    ("uploads", "Uploads", "{:.0f}"),
    ("searchable_s_per_page_p50", "Searchable s / page", "{:.2f}"),
    ("enrich_llm_calls_avg", "Enrichment LLM calls", "{:.1f}"),
]


def fmt(value, style: str) -> str:
    if value is None:
        return "–"
    if style == "ms":
        return f"{value / 1000:.2f}"
    if style == "pct":
        return f"{value * 100:.0f}%"
    return style.format(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize recorded telemetry by code version.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config).monitoring
    summary = summarize(Telemetry(cfg.path))
    if args.json:
        print(json.dumps(summary, indent=2))
        return
    if not summary:
        print("No telemetry yet. Ask some questions or upload a file first.")
        return
    versions = list(summary)
    header = "| Metric | " + " | ".join(
        f"{v} ({datetime.fromtimestamp(summary[v]['first_seen']).date()})" for v in versions) + " |"
    print(header)
    print("|" + "---|" * (len(versions) + 1))
    for key, label, style in COLUMNS:
        print(f"| {label} | " + " | ".join(fmt(summary[v][key], style) for v in versions) + " |")


if __name__ == "__main__":
    main()
