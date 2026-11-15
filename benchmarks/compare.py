"""
Compare benchmark results across versions.

    python benchmarks/compare.py                    # every dataset, latest run per version
    python benchmarks/compare.py --dataset aws_ml   # one dataset
    python benchmarks/compare.py --write            # also regenerate benchmarks/RESULTS.md

Versions are ordered by commit date; each column after the first shows the
change against the oldest version in the table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"

# (section, key, label, unit, higher_is_better)
METRICS = [
    ("ingestion", "seconds", "Ingestion time", "s", False),
    ("ingestion", "chunks", "Chunks indexed", "", None),
    ("ingestion", "llm_calls", "LLM calls while indexing", "", False),
    ("retrieval", "hit_at_1", "Hit@1", "%", True),
    ("retrieval", "hit_at_3", "Hit@3", "%", True),
    ("retrieval", "hit_at_5", "Hit@5", "%", True),
    ("retrieval", "mrr", "MRR", "", True),
    ("retrieval", "docs_avg", "Passages sent to the LLM", "", None),
    ("retrieval", "context_chars_avg", "Context characters", "", None),
    ("retrieval", "latency_ms_p50", "Retrieval latency p50", "ms", False),
    ("retrieval", "latency_ms_p95", "Retrieval latency p95", "ms", False),
    ("answers", "accuracy", "Answer accuracy", "%", True),
    ("answers", "latency_ms_p50", "Answer latency p50", "ms", False),
    ("answers", "latency_ms_p95", "Answer latency p95", "ms", False),
    ("answers", "first_token_ms_p50", "First token p50", "ms", False),
    ("answers", "llm_calls_avg", "LLM calls per question", "", False),
    ("answers", "errors", "Errors", "", False),
]


def load(dataset: str | None, mode: str) -> dict[str, list[dict]]:
    """dataset -> latest result per commit, ordered by commit date."""
    out: dict[str, list[dict]] = {}
    for path in sorted(RESULTS.glob("*/*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["mode"] != mode or (dataset and result["dataset"]["name"] != dataset):
            continue
        out.setdefault(result["dataset"]["name"], []).append(result)
    for name, results in out.items():
        latest: dict[str, dict] = {}
        for r in sorted(results, key=lambda r: r["ran_at"]):
            latest[r["version"]["commit"]] = r
        out[name] = sorted(latest.values(), key=lambda r: r["version"]["date"])
    return out


def fmt(value, unit: str) -> str:
    if value is None:
        return "–"
    if unit == "%":
        return f"{value * 100:.0f}%"
    if unit == "s":
        return f"{value:.1f} s"
    if unit == "ms":
        return f"{value:,.0f} ms"
    return f"{value:,.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else f"{value:,}"


def delta(base, value, unit: str, better) -> str:
    if base is None or value is None or base == value:
        return ""
    if unit == "%":
        diff = (value - base) * 100
        text = f"{diff:+.0f} pts"
    elif base:
        text = f"{(value - base) / base * 100:+.0f}%"
    else:
        return ""
    if better is None:
        return f" ({text})"
    good = (value > base) == better
    return f" ({text} {'✓' if good else '✗'})"


def table(results: list[dict]) -> str:
    head = "| Metric | " + " | ".join(
        f"`{r['version']['commit']}` {r['version']['subject'][:38]}" for r in results) + " |"
    lines = [head, "|" + "---|" * (len(results) + 1)]
    base = results[0]
    for section, key, label, unit, better in METRICS:
        values = [r.get(section, {}).get(key) for r in results]
        if all(v is None for v in values):
            continue
        b = base.get(section, {}).get(key)
        cells = [fmt(values[0], unit)] + [fmt(v, unit) + delta(b, v, unit, better) for v in values[1:]]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report(dataset: str | None, mode: str) -> str:
    data = load(dataset, mode)
    if not data:
        return f"No {mode} results yet. Run: python benchmarks/run.py" + (" --llm" if mode == "llm" else "")
    parts = []
    for name, results in data.items():
        r0 = results[-1]
        parts.append(f"### {name} ({mode})\n\n{r0['dataset']['description']}, "
                     f"{r0['dataset']['questions']} questions. Machine: {r0['machine']['platform']}, "
                     f"{r0['machine']['cpus']} CPUs.\n\n{table(results)}")
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark results across versions.")
    parser.add_argument("--dataset")
    parser.add_argument("--mode", choices=["offline", "llm", "all"], default="all")
    parser.add_argument("--write", action="store_true", help="regenerate benchmarks/RESULTS.md")
    args = parser.parse_args()
    modes = ["offline", "llm"] if args.mode == "all" else [args.mode]
    text = "\n\n".join(report(args.dataset, m) for m in modes)
    print(text)
    if args.write:
        header = ("# Benchmark results\n\nGenerated by `python benchmarks/compare.py --write` from the JSON files "
                  "in `benchmarks/results/`. Each column is one commit (latest run); changes are against the "
                  "oldest commit shown. ✓ marks an improvement, ✗ a regression. See `benchmarks/run.py` for "
                  "exactly what is measured.\n\n")
        (ROOT / "benchmarks" / "RESULTS.md").write_text(header + text + "\n", encoding="utf-8")
        print("\nWrote benchmarks/RESULTS.md")


if __name__ == "__main__":
    main()
