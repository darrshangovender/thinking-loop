"""Self-Consistency strategy (Wang et al., 2022).

Run Chain-of-Thought N times at temperature > 0, then take a MAJORITY VOTE
on the final answer. The idea: if the model can reach the right answer via
several different reasoning paths, the answer is more likely correct.

Empirically this is one of the strongest cheap lifts on reasoning benchmarks
(GSM8K, ARC) — and notably, it works *without* a critic. The model votes
against itself.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter

from ..budget import Budget
from ..trace import Trace
from .base import Candidate, Strategy
from .cot import SYSTEM as COT_SYSTEM, _extract_answer


def _normalise_answer(ans: str) -> str:
    """Loose normalisation for the majority vote. Strip punctuation, lowercase,
    extract last number if there is one (handles 'about 5' / '5' / '5.00')."""
    s = ans.strip().lower().rstrip(".!?")
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if nums:
        n = float(nums[-1])
        return str(int(n)) if n.is_integer() else f"{n:g}"
    return s


class SelfConsistency(Strategy):
    name = "self_consistency"

    def __init__(self, *, model: str, n: int = 5, temperature: float = 0.7):
        super().__init__(model=model)
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.temperature = temperature

    async def run(self, question: str, *, trace: Trace, budget: Budget) -> Candidate:
        trace.emit("strategy_start", self.name, n=self.n, temperature=self.temperature)

        # Fire N CoT samples in parallel — each call enforces the budget.
        async def one_sample(i: int) -> str:
            try:
                budget.check()
                return await self._llm(
                    messages=[{"role": "user", "content": question}],
                    system=COT_SYSTEM, trace=trace, budget=budget,
                    max_tokens=1024, temperature=self.temperature,
                )
            except Exception:
                return ""

        texts = await asyncio.gather(*(one_sample(i) for i in range(self.n)))
        raw_answers = [_extract_answer(t) for t in texts if t]
        if not raw_answers:
            trace.emit("strategy_end", self.name, answer="", note="all samples failed")
            return Candidate(strategy=self.name, answer="", reasoning="all samples failed")

        # Majority vote on normalised form, pick the original surface form with most votes.
        normalised_to_original: dict[str, list[str]] = {}
        for raw in raw_answers:
            normalised_to_original.setdefault(_normalise_answer(raw), []).append(raw)
        counts = Counter({k: len(v) for k, v in normalised_to_original.items()})
        winning_norm, votes = counts.most_common(1)[0]
        winning_answer = normalised_to_original[winning_norm][0]
        confidence = votes / len(raw_answers)

        trace.emit("strategy_end", self.name, answer=winning_answer, votes=votes, samples=len(raw_answers))
        return Candidate(
            strategy=self.name,
            answer=winning_answer,
            reasoning=f"Majority vote: {votes}/{len(raw_answers)} → {winning_answer}",
            intermediate={"vote_counts": dict(counts), "n_samples": len(raw_answers), "all_answers": raw_answers},
            self_score=confidence,
        )