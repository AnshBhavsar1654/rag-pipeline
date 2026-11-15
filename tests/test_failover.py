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
