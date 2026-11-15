"""LLM failover: rate-limited models cool down; errors become friendly text."""

from __future__ import annotations

import pytest
from langchain_core.language_models import FakeListChatModel

from src.generation.failover import FailoverChatModel, classify_error, friendly_error


class RateLimited(FakeListChatModel):
    calls: int = 0

    def _call(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

    def _stream(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")


def make(primary, backup):
    return FailoverChatModel(models=[primary, backup], names=["primary", "backup"], cooldown_seconds=60)


def test_rate_limit_moves_to_backup_and_cools_primary_down():
    primary = RateLimited(responses=["unused"])
    llm = make(primary, FakeListChatModel(responses=["from backup"]))
    assert llm.invoke("hi").content == "from backup"
    assert llm.invoke("again").content == "from backup"
    assert primary.calls == 1          # second call skipped the cooling primary
    assert llm.last_used == "backup"


def test_streaming_fails_over_before_any_text():
    llm = make(RateLimited(responses=["x"]), FakeListChatModel(responses=["streamed"]))
    assert "".join(c.content for c in llm.stream("hi")) == "streamed"


def test_all_models_failing_raises_last_error():
    llm = make(RateLimited(responses=["x"]), RateLimited(responses=["y"]))
    with pytest.raises(RuntimeError, match="429"):
        llm.invoke("hi")


def test_errors_are_classified_and_made_friendly():
    assert classify_error(RuntimeError("429 Too Many Requests")) == "rate_limit"
    assert classify_error(RuntimeError("401 invalid api key")) == "auth"
    assert classify_error(RuntimeError("503 model overloaded")) == "unavailable"
    message = friendly_error(RuntimeError("ResourceExhausted: 429 quota"))
    assert "429" not in message and "too many requests" in message.lower()


def test_provider_retry_delay_sets_the_cooldown():
    from src.generation.failover import retry_after_seconds
    assert retry_after_seconds(RuntimeError("'retryDelay': '45683s'")) == 45683
    assert retry_after_seconds(RuntimeError("Please retry in 12h41m23.5s.")) == 12 * 3600 + 41 * 60 + 23.5
    assert retry_after_seconds(RuntimeError("Please try again in 7.66s.")) == 7.66
    assert retry_after_seconds(RuntimeError("boom")) is None


def test_daily_quota_pauses_the_model_and_reports_why():
    class Daily(FakeListChatModel):
        def _call(self, *a, **k):
            raise RuntimeError("429 RESOURCE_EXHAUSTED requests per day limit. Please retry in 2h0m0s.")

    llm = make(Daily(responses=["x"]), FakeListChatModel(responses=["ok"]))
    assert llm.invoke("hi").content == "ok"
    primary = llm.status()[0]
    assert primary["paused_for"] > 7000 and primary["reason"] == "daily free quota used up"
