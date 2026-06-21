"""ThinkingLoop — the main orchestrator.

Runs strategies in parallel, adjudicates candidates, calibrates confidence,
returns Answer with full trace. Hard budget guards throughout.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Sequence

from .adjudicator import Adjudicator, AdjudicationResult
from .budget import Budget, BudgetExceeded
from .confidence import calibrate
from .strategies.base import Candidate, Strategy
from .trace import Trace


@dataclass
class Answer:
    final: str
    confidence: float                  # 0..1, calibrated
    winning_strategy: str
    candidates: list[Candidate] = field(default_factory=list)
    adjudication: AdjudicationResult | None = None
    trace: Trace = field(default_factory=Trace)


class ThinkingLoop:
    def __init__(
        self,
        strategies: Sequence[Strategy],
        *,
        adjudicator_model: str = "claude-opus-4-7",
        budget: Budget | None = None,
    ):
        if not strategies:
            raise ValueError("at least one strategy required")
        self.strategies = list(strategies)
        self.adjudicator = Adjudicator(model=adjudicator_model)
        self.budget = budget or Budget()

    def solve(self, question: str) -> Answer:
        """Sync entry point. Runs the async loop internally — cheap for callers."""
        return asyncio.run(self.asolve(question))

    async def asolve(self, question: str) -> Answer:
        trace = Trace()
        self.budget.reset()

        # 1. Fire all strategies in parallel.
        async def run_one(s: Strategy) -> Candidate | None:
            try:
                return await s.run(question, trace=trace, budget=self.budget)
            except BudgetExceeded as e:
                trace.emit("budget_exceeded", s.name, reason=str(e))
                return None
            except Exception as e:
                trace.emit("error", s.name, error=str(e)[:300])
                return None

        results = await asyncio.gather(*(run_one(s) for s in self.strategies))
        candidates = [c for c in results if c is not None and c.answer]

        if not candidates:
            return Answer(final="", confidence=0.0, winning_strategy="(none)", trace=trace)

        # 2. Adjudicate.
        try:
            adj = await self.adjudicator.judge(question, candidates, trace=trace, budget=self.budget)
        except Exception as e:
            trace.emit("error", "adjudicator", error=str(e)[:300])
            # Fall back: pick the first candidate from the most expensive strategy.
            winner = candidates[0]
            return Answer(
                final=winner.answer, confidence=0.5, winning_strategy=winner.strategy,
                candidates=candidates, adjudication=None, trace=trace,
            )

        # 3. Calibrate confidence.
        confidence = calibrate(candidates, adj)
        winner = candidates[adj.winner_index]

        return Answer(
            final=winner.answer,
            confidence=confidence,
            winning_strategy=winner.strategy,
            candidates=candidates,
            adjudication=adj,
            trace=trace,
        )