"""
Benchmark any version of this pipeline on a fixed question set.

    python benchmarks/run.py                          # working tree, offline, every dataset
    python benchmarks/run.py --ref ac2f37a --ref HEAD # check out each commit and run it
    python benchmarks/run.py --dataset sample --llm   # also generate answers (uses API quota)

Each run writes benchmarks/results/<dataset>/<date>_<commit>_<mode>.json;
compare them with ``python benchmarks/compare.py``.

What is measured
----------------
Ingestion: wall time to index the dataset's corpus (embedding-model load
excluded), chunks produced, LLM calls spent.
Retrieval: for each question, whether a returned passage contains the
known evidence text: hit@1/3/5, MRR (1 / rank of the first relevant
passage), passages and characters handed to the LLM, and latency.
Answers (--llm only): accuracy against expected keywords, end-to-end and
first-token latency, LLM calls per question, errors.

Offline mode (the default) makes no LLM calls: query translation is set to
"simple", the LLM-only stages (graph extraction, RAPTOR, Text-to-SQL/Cypher,
self-query, Self-RAG, web search) are switched off, and the model is replaced
by a stand-in that refuses, so any attempted call is counted. That isolates
chunking, indexing, hybrid search, re-ranking and parent documents, which
can then be compared across versions without spending quota.

Older commits are driven only through APIs every version has
(``RAGPipeline(config)``, ``ingest()``, ``retrieve()``, ``query()``), with
that version's own config.yaml plus the overrides above, so each version is
measured with its own defaults.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "benchmarks" / "datasets"
RESULTS = ROOT / "benchmarks" / "results"


# ---------------------------------------------------------------------------
# Parent: pick versions, check them out, run the child for each
# ---------------------------------------------------------------------------

def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def commit_info(ref: str) -> dict:
    if ref == "WORKTREE":
        sha = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
        return {"ref": "WORKTREE", "commit": sha[:7] + ("+dirty" if dirty else ""),
                "subject": "working tree" + (" (uncommitted changes)" if dirty else ""),
                "date": datetime.now().isoformat(timespec="seconds")}
    sha = git("rev-parse", ref)
    subject, date = git("log", "-1", "--format=%s%x1f%cI", sha).split("\x1f")
    return {"ref": ref, "commit": sha[:7], "subject": subject, "date": date}


def run_version(ref: str, dataset: Path, llm: bool, workdir: Path, label: str | None = None) -> Path | None:
    info = commit_info(ref)
    if label:   # an experiment (e.g. a config change): keep it apart from the plain commit
        info.update(label=label, subject=label, commit=f"{info['commit']}/{label}")
    tree = None
    if ref == "WORKTREE":
        code_dir = ROOT
    else:
        sha = info["commit"].split("/")[0]
        code_dir = workdir / f"tree-{sha}"
        if not code_dir.exists():
            git("worktree", "add", "--detach", str(code_dir), sha)
        tree = code_dir
    mode = "llm" if llm else "offline"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in info["commit"])
    out = RESULTS / dataset.stem / f"{datetime.now():%Y%m%d-%H%M%S}_{safe}_{mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--code-dir", str(code_dir),
           "--dataset-file", str(dataset), "--out", str(out), "--version-json", json.dumps(info)]
    if llm:
        cmd.append("--llm")
    print(f"\n=== {info['commit']} ({info['subject']}) on {dataset.stem}, {mode}")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    scratch = workdir / f"run-{safe}-{dataset.stem}"
    scratch.mkdir(parents=True, exist_ok=True)
    started = time.time()
    proc = subprocess.run(cmd, cwd=scratch, env=env)
    if tree is not None:
        subprocess.run(["git", "worktree", "remove", "--force", str(tree)], cwd=ROOT, capture_output=True)
    if proc.returncode != 0 or not out.exists():
        print(f"!!! {info['commit']} failed (exit {proc.returncode}) after {time.time() - started:.0f}s")
        return None
    print(f"--> {out.relative_to(ROOT)} ({time.time() - started:.0f}s)")
    return out


def parent(args) -> None:
    import yaml

    datasets = sorted(DATASETS.glob("*.yaml"))
    if args.dataset:
        datasets = [d for d in datasets if d.stem in args.dataset]
    runnable = []
    for d in datasets:
        spec = yaml.safe_load(d.read_text(encoding="utf-8"))
        missing = [c for c in spec["corpus"] if not (ROOT / c).exists()]
        if missing:
            print(f"Skipping {d.stem}: missing {', '.join(missing)}")
        else:
            runnable.append(d)
    refs = args.ref or ["WORKTREE"]
    workdir = Path(tempfile.mkdtemp(prefix="rag-bench-"))
    written = []
    try:
        for dataset in runnable:
            for ref in refs:
                path = run_version(ref, dataset, args.llm, workdir, args.label)
                if path:
                    written.append(path)
    finally:
        subprocess.run(["git", "worktree", "prune"], cwd=ROOT, capture_output=True)
        shutil.rmtree(workdir, ignore_errors=True)
    if written:
        print("\nCompare with:  python benchmarks/compare.py" + (f" --dataset {args.dataset[0]}" if args.dataset else ""))


# ---------------------------------------------------------------------------
# Child: runs inside one version's code
# ---------------------------------------------------------------------------

def norm(text: str) -> str:
    """Case-, ligature- and whitespace-insensitive form for evidence matching."""
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def deep_set(cfg: dict, dotted: str, value) -> None:
    node = cfg
    *path, last = dotted.split(".")
    for key in path:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            return
    node[last] = value


def pct(values: list[float], q: float) -> float | None:
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 1)


def make_llm_wrappers():
    from langchain_core.language_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from pydantic import ConfigDict

    class Counting(BaseChatModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        inner: BaseChatModel | None = None
        calls: int = 0

        @property
        def _llm_type(self) -> str:
            return "counting"

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            self.calls += 1
            if self.inner is None:
                raise RuntimeError("offline benchmark: the LLM is disabled")
            return ChatResult(generations=[ChatGeneration(message=self.inner.invoke(messages, stop=stop, **kw))])

        def _stream(self, messages, stop=None, run_manager=None, **kw):
            self.calls += 1
            if self.inner is None:
                raise RuntimeError("offline benchmark: the LLM is disabled")
            for chunk in self.inner.stream(messages, stop=stop, **kw):
                yield ChatGenerationChunk(message=chunk)

    return Counting


def child(args) -> None:
    import yaml

    code_dir = Path(args.code_dir).resolve()
    sys.path.insert(0, str(code_dir))
    from src.config import RAGConfig
    from src.pipeline.rag_pipeline import RAGPipeline

    spec = yaml.safe_load(Path(args.dataset_file).read_text(encoding="utf-8"))
    raw = yaml.safe_load((code_dir / "config" / "config.yaml").read_text(encoding="utf-8")) or {}
    store = Path(tempfile.mkdtemp(prefix="stores-"))
    overrides = {
        "documents.sources": [str(ROOT / c) for c in spec["corpus"]],
        "vectorstore.persist_directory": str(store / "vectors"),
        "relational_db.path": str(store / "relational.sqlite"),
        "graph_db.path": str(store / "graph.json"),
        "indexing.docstore_path": str(store / "docstore.sqlite"),
        "monitoring.enabled": False,
        "chat.rewrite_followups": False,
    }
    if not args.llm:
        overrides.update({
            "retrieval.strategy": "simple",
            "retrieval.crag.web_search": False,
            "graph_db.enabled": False,
            "indexing.raptor.enabled": False,
            "indexing.multi_representation": raw.get("indexing", {}).get("multi_representation", "none")
            if raw.get("indexing", {}).get("multi_representation") != "summary" else "parent_document",
            "query_construction.text_to_sql": False,
            "query_construction.text_to_cypher": False,
            "query_construction.self_query": False,
            "routing.logical": False,
            "generation.self_rag.enabled": False,
        })
    for key, value in overrides.items():
        deep_set(raw, key, value)
    config = RAGConfig(**raw)          # unknown keys are ignored by older versions
    pipeline = RAGPipeline(config)

    t = time.perf_counter()
    pipeline.embeddings
    model_load_s = time.perf_counter() - t

    Counting = make_llm_wrappers()
    llm = Counting(inner=None if not args.llm else pipeline.llm)
    pipeline._llm = llm

    t = time.perf_counter()
    pipeline.ingest()
    ingest_s = time.perf_counter() - t
    chunks = pipeline._vectorstore._collection.count()
    ingest_calls = llm.calls

    questions = spec["questions"]
    pipeline.retrieve("warm up the reranker and caches")
    rows = []
    for q in questions:
        evidence = [norm(e) for e in q["evidence"]]
        calls_before = llm.calls
        t = time.perf_counter()
        docs = pipeline.retrieve(q["question"])
        retrieve_ms = (time.perf_counter() - t) * 1000
        rank = next((i for i, d in enumerate(docs, 1) if any(e in norm(d.page_content) for e in evidence)), None)
        row = {
            "question": q["question"], "rank": rank, "docs": len(docs),
            "context_chars": sum(len(d.page_content) for d in docs),
            "retrieve_ms": round(retrieve_ms, 1), "retrieve_llm_calls": llm.calls - calls_before,
        }
        if args.llm:
            row.update(answer_question(pipeline, llm, q))
        rows.append(row)
        print(f"  rank={rank!s:>4}  {retrieve_ms:7.0f} ms  {q['question']}")

    n = len(rows)
    result = {
        "version": json.loads(args.version_json),
        "dataset": {"name": spec["name"], "description": spec.get("description", ""), "questions": n,
                    "corpus": spec["corpus"]},
        "mode": "llm" if args.llm else "offline",
        "machine": {"platform": platform.platform(), "python": platform.python_version(),
                    "cpus": os.cpu_count()},
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "ingestion": {"seconds": round(ingest_s, 2), "chunks": chunks, "llm_calls": ingest_calls,
                      "model_load_seconds": round(model_load_s, 2)},
        "retrieval": {
            "hit_at_1": round(sum(1 for r in rows if r["rank"] and r["rank"] <= 1) / n, 3),
            "hit_at_3": round(sum(1 for r in rows if r["rank"] and r["rank"] <= 3) / n, 3),
            "hit_at_5": round(sum(1 for r in rows if r["rank"] and r["rank"] <= 5) / n, 3),
            "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3),
            "docs_avg": round(statistics.mean(r["docs"] for r in rows), 2),
            "context_chars_avg": round(statistics.mean(r["context_chars"] for r in rows)),
            "latency_ms_p50": pct([r["retrieve_ms"] for r in rows], 0.5),
            "latency_ms_p95": pct([r["retrieve_ms"] for r in rows], 0.95),
            "llm_calls_avg": round(statistics.mean(r["retrieve_llm_calls"] for r in rows), 2),
        },
        "questions": rows,
    }
    if args.llm:
        answered = [r for r in rows if not r.get("error")]
        result["answers"] = {
            "accuracy": round(sum(1 for r in rows if r.get("correct")) / n, 3),
            "errors": n - len(answered),
            "latency_ms_p50": pct([r["answer_ms"] for r in answered], 0.5),
            "latency_ms_p95": pct([r["answer_ms"] for r in answered], 0.95),
            "first_token_ms_p50": pct([r["first_token_ms"] for r in answered if r.get("first_token_ms")], 0.5),
            "llm_calls_avg": round(statistics.mean(r["answer_llm_calls"] for r in rows), 2),
        }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


def answer_question(pipeline, llm, q: dict) -> dict:
    calls_before = llm.calls
    started = time.perf_counter()
    first_token_ms = None
    try:
        if hasattr(pipeline, "run"):            # newer versions stream events
            answer = ""
            for event in pipeline.run(q["question"]):
                if event.kind == "token" and first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                if event.kind == "done":
                    answer = event.answer
        else:
            answer, _ = pipeline.query(q["question"])
        error = None
    except Exception as e:  # noqa: BLE001
        answer, error = "", f"{type(e).__name__}: {str(e)[:200]}"
    text = norm(answer)
    correct = bool(answer) and all(any(norm(alt) in text for alt in group) for group in q["answer"])
    return {
        "answer": answer, "correct": correct, "error": error,
        "answer_ms": round((time.perf_counter() - started) * 1000, 1),
        "first_token_ms": round(first_token_ms, 1) if first_token_ms else None,
        "answer_llm_calls": llm.calls - calls_before,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", action="append", help="git ref to benchmark (repeatable); default: working tree")
    parser.add_argument("--dataset", action="append", help="dataset name (repeatable); default: all available")
    parser.add_argument("--llm", action="store_true", help="also generate and score answers (uses API quota)")
    parser.add_argument("--label", help="name this run as an experiment (e.g. after editing config.yaml)")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--code-dir", help=argparse.SUPPRESS)
    parser.add_argument("--dataset-file", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    parser.add_argument("--version-json", help=argparse.SUPPRESS)
    args = parser.parse_args()
    child(args) if args.child else parent(args)


if __name__ == "__main__":
    main()
