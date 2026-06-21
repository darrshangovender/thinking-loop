"""Structured trace events. Every strategy run + LLM call emits one of these."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

EventKind = Literal["strategy_start", "strategy_end", "llm_call", "adjudicator_call", "budget_exceeded", "error"]


@dataclass
class TraceEvent:
    kind: EventKind
    actor: str
    ts: datetime
    duration_ms: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0


class Trace:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self._spans: dict[str, float] = {}

    def emit(self, kind: EventKind, actor: str, **payload: Any) -> None:
        self.events.append(TraceEvent(kind=kind, actor=actor, ts=datetime.now(timezone.utc), payload=payload))

    def emit_llm(self, actor: str, *, model: str, tokens_in: int, tokens_out: int, cost_usd: float | None, duration_ms: int, **extra: Any) -> None:
        self.events.append(TraceEvent(
            kind="llm_call", actor=actor, ts=datetime.now(timezone.utc),
            duration_ms=duration_ms, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd,
            payload={"model": model, **extra},
        ))

    def start_span(self, key: str) -> None:
        self._spans[key] = time.perf_counter()

    def end_span(self, key: str) -> int:
        t0 = self._spans.pop(key, time.perf_counter())
        return int((time.perf_counter() - t0) * 1000)

    def llm_calls(self) -> list[TraceEvent]:
        return [e for e in self.events if e.kind == "llm_call"]

    def for_actor(self, actor: str) -> list[TraceEvent]:
        return [e for e in self.events if e.actor == actor]

    def total_cost_usd(self) -> float:
        return sum(e.cost_usd or 0.0 for e in self.events)

    def total_tokens(self) -> tuple[int, int]:
        return (sum(e.tokens_in for e in self.events), sum(e.tokens_out for e in self.events))

    def summary(self) -> str:
        ti, to = self.total_tokens()
        n_strats = len(set(e.actor for e in self.events if e.kind == "strategy_end"))
        return (
            f"{n_strats} strategies, {len(self.llm_calls())} LLM calls, "
            f"tokens={ti}+{to}, ${self.total_cost_usd():.4f}"
        )