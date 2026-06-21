"""Direct strategy — single LLM call. The baseline.

This exists so the adjudicator has a control to compare against. Often the
direct answer is correct and cheap; the loop should not always prefer slower
strategies just because they're slower.
"""

from __future__ import annotations

from ..budget import Budget
from ..trace import Trace
from .base import Candidate, Strategy


SYSTEM = "Answer the question directly and concisely. State the final answer on the last line, prefixed with 'Answer: '."


class Direct(Strategy):
    name = "direct"

    async def run(self, question: str, *, trace: Trace, budget: Budget) -> Candidate:
        trace.emit("strategy_start", self.name)
        try:
            text = await self._llm(
                messages=[{"role": "user", "content": question}],
                system=SYSTEM, trace=trace, budget=budget,
                max_tokens=512, temperature=0.2,
            )
            answer = _extract_answer(text)
            trace.emit("strategy_end", self.name, answer=answer)
            return Candidate(strategy=self.name, answer=answer, reasoning=text)
        except Exception as e:
            trace.emit("error", self.name, error=str(e)[:200])
            raise


def _extract_answer(text: str) -> str:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.lower().startswith("answer:"):
            return line.split(":", 1)[1].strip()
    return text.strip().splitlines()[-1] if text.strip() else ""