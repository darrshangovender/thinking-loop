"""Chain-of-Thought strategy (Wei et al., 2022).

Force the model to reason step-by-step before giving the final answer.
The trick is the prompt: 'think step by step' and a strict answer format.
"""

from __future__ import annotations

from ..budget import Budget
from ..trace import Trace
from .base import Candidate, Strategy


SYSTEM = """\
Reason step by step before answering. Format your response as:

THOUGHT: <numbered chain of intermediate reasoning steps>
Answer: <final answer, one line>

Be thorough in THOUGHT but concise on the final Answer line.
"""


class ChainOfThought(Strategy):
    name = "chain_of_thought"

    async def run(self, question: str, *, trace: Trace, budget: Budget) -> Candidate:
        trace.emit("strategy_start", self.name)
        text = await self._llm(
            messages=[{"role": "user", "content": question}],
            system=SYSTEM, trace=trace, budget=budget,
            max_tokens=1024, temperature=0.3,
        )
        answer = _extract_answer(text)
        trace.emit("strategy_end", self.name, answer=answer)
        return Candidate(
            strategy=self.name, answer=answer, reasoning=text,
            intermediate={"chain": _extract_thought(text)},
        )


def _extract_answer(text: str) -> str:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.lower().startswith("answer:"):
            return line.split(":", 1)[1].strip()
    return text.strip().splitlines()[-1] if text.strip() else ""


def _extract_thought(text: str) -> str:
    for marker in ("THOUGHT:", "Thought:"):
        if marker in text:
            after = text.split(marker, 1)[1]
            return after.split("Answer:")[0].strip()
    return text