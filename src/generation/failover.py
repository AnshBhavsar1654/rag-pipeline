"""
Generation — LLM failover and friendly errors

``FailoverChatModel`` tries the primary model and, when it fails, moves to
the next one straight away instead of waiting through retries. A model that
reports a rate limit or outage is put on cooldown for a while, so the
following calls (query planning, background indexing, the answer) go
directly to the fallback instead of hitting the limit again.

``friendly_error`` turns provider exceptions into a short message the UI
can show to a person, instead of a raw API error.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict, PrivateAttr


_RATE_LIMIT_MARKERS = ("429", "rate limit", "rate_limit", "ratelimit", "quota", "resource_exhausted",
                       "resourceexhausted", "too many requests", "tokens per minute", "requests per minute")
_UNAVAILABLE_MARKERS = ("503", "502", "500", "overloaded", "unavailable", "timeout", "timed out",
                        "connection", "deadline")
_AUTH_MARKERS = ("401", "403", "api key", "api_key", "permission", "unauthorized", "authentication")


def classify_error(error: BaseException) -> str:
    """rate_limit | unavailable | auth | other"""
    text = f"{type(error).__name__} {error}".lower()
    if any(m in text for m in _RATE_LIMIT_MARKERS):
        return "rate_limit"
    if any(m in text for m in _AUTH_MARKERS):
        return "auth"
    if any(m in text for m in _UNAVAILABLE_MARKERS):
        return "unavailable"
    return "other"


_MAX_COOLDOWN = 24 * 3600


def retry_after_seconds(error: BaseException) -> float | None:
    """How long the provider asked us to wait, if the error says so.

    Understands e.g. "retryDelay': '45683s'", "Please retry in 12h41m23.5s"
    and "Please try again in 7.66s"."""
    text = str(error)
    match = re.search(r"retry_?delay['\"]?\s*[:=]\s*['\"]?([\d.]+)s", text, re.I)
    if match:
        return min(float(match.group(1)), _MAX_COOLDOWN)
    match = re.search(r"(?:retry|try again) in ((?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+s)?)", text, re.I)
    if match and match.group(1):
        parts = re.findall(r"([\d.]+)([hms])", match.group(1))
        seconds = sum(float(v) * {"h": 3600, "m": 60, "s": 1}[u] for v, u in parts)
        return min(seconds, _MAX_COOLDOWN) if seconds else None
    return None


def friendly_error(error: BaseException) -> str:
    """A short, human message for an error raised while answering."""
    kind = classify_error(error)
    if kind == "rate_limit":
        return ("The AI service is handling too many requests right now. "
                "Please wait a few seconds and try again.")
    if kind == "auth":
        return "The AI service rejected the API key. Check the keys in your .env file."
    if kind == "unavailable":
        return "The AI service is temporarily unavailable. Please try again in a moment."
    return "Something went wrong while answering. Please try again."


class FailoverChatModel(BaseChatModel):
    """Chat model that falls through a list of models, cooling down failing ones."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    models: list[BaseChatModel]
    names: list[str]
    cooldown_seconds: float = 60.0

    _cooldown_until: dict[int, float] = PrivateAttr(default_factory=dict)
    _cooldown_reason: dict[int, str] = PrivateAttr(default_factory=dict)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _last_used: str = PrivateAttr(default="")

    @property
    def _llm_type(self) -> str:
        return "failover"

    @property
    def last_used(self) -> str:
        """Name of the model that served the most recent call."""
        return self._last_used

    def _order(self) -> list[int]:
        """Models not cooling down, in priority order; then the cooling ones, soonest first."""
        now = time.monotonic()
        with self._lock:
            ready = [i for i in range(len(self.models)) if self._cooldown_until.get(i, 0) <= now]
            cooling = sorted(
                (i for i in range(len(self.models)) if i not in ready),
                key=lambda i: self._cooldown_until[i],
            )
        return ready + cooling

    def status(self) -> list[dict]:
        """Each model, and for how long it is paused (seconds) and why."""
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "name": name,
                    "paused_for": max(0, int(self._cooldown_until.get(i, 0) - now)),
                    "reason": self._cooldown_reason.get(i, "") if self._cooldown_until.get(i, 0) > now else "",
                }
                for i, name in enumerate(self.names)
            ]

    def _failed(self, index: int, error: BaseException) -> None:
        kind = classify_error(error)
        print(f"[LLM] {self.names[index]} failed ({kind}): {str(error)[:160]}")
        if kind in ("rate_limit", "unavailable"):
            # Honour the provider's own retry delay (a daily quota can mean hours).
            wait = max(self.cooldown_seconds, retry_after_seconds(error) or 0)
            daily = "per day" in str(error).lower() or "perday" in str(error).lower()
            with self._lock:
                self._cooldown_until[index] = time.monotonic() + wait
                self._cooldown_reason[index] = (
                    "daily free quota used up" if daily
                    else "rate limited" if kind == "rate_limit" else "temporarily unavailable"
                )
            print(f"[LLM] Skipping {self.names[index]} for {wait:.0f}s")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_error: BaseException | None = None
        for i in self._order():
            try:
                message = self.models[i].invoke(messages, stop=stop, **kwargs)
            except Exception as e:  # noqa: BLE001 - every failure moves to the next model
                self._failed(i, e)
                last_error = e
                continue
            self._last_used = self.names[i]
            return ChatResult(generations=[ChatGeneration(message=message)])
        raise last_error  # type: ignore[misc]

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        last_error: BaseException | None = None
        for i in self._order():
            started = False
            try:
                for chunk in self.models[i].stream(messages, stop=stop, **kwargs):
                    if not started:
                        self._last_used = self.names[i]
                        started = True
                    if run_manager and isinstance(chunk.content, str):
                        run_manager.on_llm_new_token(chunk.content)
                    yield ChatGenerationChunk(message=chunk)
                return
            except Exception as e:  # noqa: BLE001
                self._failed(i, e)
                if started:
                    raise  # part of the answer is out; don't mix two models' text
                last_error = e
        raise last_error  # type: ignore[misc]
