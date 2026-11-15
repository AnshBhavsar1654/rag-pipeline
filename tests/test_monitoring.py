"""Telemetry: questions and uploads are recorded with LLM call counts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from langchain_core.language_models import FakeListChatModel

from src.monitoring.meter import Meter, MeteredChatModel, iterate_metered, run_metered, submit
from src.monitoring.telemetry import code_version, percentile, summarize
from tests.test_pipeline import config, embeddings, make_pipeline  # noqa: F401


def test_meter_counts_calls_made_in_worker_threads():
    llm = MeteredChatModel(model=FakeListChatModel(responses=["a"]))
    meter = Meter()

    def work():
        with ThreadPoolExecutor(2) as pool:
            for f in [submit(pool, llm.invoke, "x") for _ in range(3)]:
                f.result()
        llm.invoke("y")

    run_metered(meter, work)
    assert meter.llm_calls == 4


def test_meter_survives_a_generator_resumed_from_other_threads():
    llm = MeteredChatModel(model=FakeListChatModel(responses=["a"]))
    meter = Meter()

    def steps():
        for _ in range(3):
            llm.invoke("x")
            yield 1

    gen = iterate_metered(meter, steps())
    with ThreadPoolExecutor(3) as pool:          # each step on a different thread
        for _ in range(3):
            pool.submit(next, gen).result()
    assert meter.llm_calls == 3


def test_questions_and_uploads_are_recorded(config, embeddings, tmp_path):
    pipeline = make_pipeline(config, embeddings, llm_responses=["It weighs 340 grams [1]."])
    pipeline._llm = MeteredChatModel(model=pipeline._llm)
    pipeline.ingest()
    pipeline.query("How heavy is the Zorblax-9?")

    (q,) = pipeline.telemetry.rows("queries")
    assert q["version"] == code_version() and q["error"] is None
    assert q["llm_calls"] == 1 and q["documents"] >= 1 and q["total_ms"] >= 0
    assert '"retrieval"' in q["stage_ms"]
    (ing,) = pipeline.telemetry.rows("ingestions")
    assert ing["chunks"] > 0 and ing["searchable_s"] >= 0 and ing["files"]

    summary = summarize(pipeline.telemetry)[code_version()]
    assert summary["questions"] == 1 and summary["llm_calls_avg"] == 1


def test_percentile():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([], 0.5) is None
