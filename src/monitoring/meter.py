"""
Monitoring — LLM usage meter

Counts the LLM calls, failovers and tokens spent on one unit of work (a
question, an upload). The active meter lives in a context variable;
``MeteredChatModel`` adds to it on every call, and the pipeline copies the
context into its worker threads so parallel calls are counted too.
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import Executor, Future
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict


@dataclass
class Meter:
    llm_calls: int = 0
    failovers: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, llm_calls: int = 0, failovers: int = 0, input_tokens: int = 0, output_tokens: int = 0) -> None:
        with self._lock:
            self.llm_calls += llm_calls
            self.failovers += failovers
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("_lock", None)
        return d


_current: contextvars.ContextVar[Meter | None] = contextvars.ContextVar("rag_meter", default=None)


def current_meter() -> Meter | None:
    return _current.get()


def record(**counts: int) -> None:
    meter = _current.get()
    if meter is not None:
        meter.add(**counts)


def submit(pool: Executor, fn: Callable, *args) -> Future:
    """``pool.submit`` that carries the active meter into the worker thread."""
    return pool.submit(contextvars.copy_context().run, fn, *args)


def run_metered(meter: Meter, fn: Callable, *args):
    """Run *fn* with *meter* active."""
    ctx = contextvars.copy_context()
    ctx.run(_current.set, meter)
    return ctx.run(fn, *args)


def iterate_metered(meter: Meter, gen: Generator):
    """Drive *gen* with *meter* active at every step, whichever thread resumes it.

    (A web server may resume a streaming generator from a different thread
    each time; running each step inside one fixed context keeps the meter.)"""
    ctx = contextvars.copy_context()
    ctx.run(_current.set, meter)
    while True:
        try:
            item = ctx.run(next, gen)
        except StopIteration as stop:
            return stop.value
        yield item


def _usage(message) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)


class MeteredChatModel(BaseChatModel):
    """Counts every call made through *model* on the active meter."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: BaseChatModel

    @property
    def _llm_type(self) -> str:
        return "metered"

    def __getattr__(self, name: str) -> Any:
        # Pass through helpers of the wrapped model (e.g. FailoverChatModel.status).
        if name.startswith("_") or name in type(self).model_fields:
            raise AttributeError(name)
        return getattr(self.model, name)

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        record(llm_calls=1)
        message = self.model.invoke(messages, stop=stop, **kwargs)
        tokens_in, tokens_out = _usage(message)
        record(input_tokens=tokens_in, output_tokens=tokens_out)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(self, messages: list[BaseMessage], stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        record(llm_calls=1)
        for chunk in self.model.stream(messages, stop=stop, **kwargs):
            tokens_in, tokens_out = _usage(chunk)
            if tokens_in or tokens_out:
                record(input_tokens=tokens_in, output_tokens=tokens_out)
            if run_manager and isinstance(chunk.content, str):
                run_manager.on_llm_new_token(chunk.content)
            yield ChatGenerationChunk(message=chunk)
