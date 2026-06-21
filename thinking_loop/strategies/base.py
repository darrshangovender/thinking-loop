"""Strategy base class. Implement run() to add a new strategy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..budget import Budget
from ..llm import LLM
from ..trace import Trace


@dataclass
class Candidate:
    """What every strategy returns. The adjudicator scores Candidate.answer."""
    strategy: str
    answer: str
    reasoning: str = ""                # human-readable trace of the strategy's thinking
    intermediate: dict[str, Any] = field(default_factory=dict)  # strategy-specific data the adjudicator can inspect
    self_score: float | None = None    # strategy's own confidence (if it produces one)


class Strategy(ABC):
    """Base class. Subclass + implement `run`. Strategies are stateless across calls."""

    name: str = "strategy"

    def __init__(self, *, model: str):
        self.model = model
        self.llm = LLM(model)

    @abstractmethod
    async def run(self, question: str, *, trace: Trace, budget: Budget) -> Candidate:
        ...

    async def _llm(self, *, messages: list[dict], system: str, trace: Trace, budget: Budget, max_tokens: int = 1024, temperature: float = 0.4) -> str:
        """Helper: LLM call with trace + budget bookkeeping. All strategies use this."""
        budget.check()
        resp = await self.llm.chat(messages, system=system, max_tokens=max_tokens, temperature=temperature)
        trace.emit_llm(self.name, model=resp.model, tokens_in=resp.tokens_in, tokens_out=resp.tokens_out, cost_usd=resp.cost_usd, duration_ms=resp.duration_ms)
        budget.add(tokens_in=resp.tokens_in, tokens_out=resp.tokens_out, cost_usd=resp.cost_usd)
        return resp.content