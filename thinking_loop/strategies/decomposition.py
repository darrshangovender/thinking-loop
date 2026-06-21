"""Decomposition strategy (Press et al., 2022; aka 'measuring and narrowing the compositionality gap').

Break the question into ordered sub-questions, answer each in turn (with
earlier answers visible), then compose the final answer. Two LLM calls:
one to decompose + answer each sub-question, one to compose.

The cheap version uses a single LLM call with a structured prompt; the
expensive version uses one call per sub-question. We use the single-call
form because it's faster and the empirical lift over multi-call is marginal
for questions with ≤4 sub-questions (which is most questions).
"""

from __future__ import annotations

import re

from ..budget import Budget
from ..trace import Trace
from .base import Candidate, Strategy


SYSTEM = """\
You are decomposing and answering hard questions. Format:

DECOMPOSITION:
1. <sub-question>
   <its answer>
2. <sub-question>
   <its answer>
...

COMPOSITION: <how the sub-answers combine into the final answer>
Answer: <final answer, one line>

Aim for 2-5 sub-questions. Each sub-question must be answerable on its own.
"""


class Decomposition(Strategy):
    name = "decomposition"

    async def run(self, question: str, *, trace: Trace, budget: Budget) -> Candidate:
        trace.emit("strategy_start", self.name)
        text = await self._llm(
            messages=[{"role": "user", "content": question}],
            system=SYSTEM, trace=trace, budget=budget,
            max_tokens=1500, temperature=0.3,
        )
        answer = _extract_answer(text)
        sub_qs = _extract_sub_questions(text)
        trace.emit("strategy_end", self.name, answer=answer, sub_question_count=len(sub_qs))
        return Candidate(
            strategy=self.name, answer=answer, reasoning=text,
            intermediate={"sub_questions": sub_qs},
        )


def _extract_answer(text: str) -> str:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.lower().startswith("answer:"):
            return line.split(":", 1)[1].strip()
    return text.strip().splitlines()[-1] if text.strip() else ""


def _extract_sub_questions(text: str) -> list[str]:
    """Pull lines that look like '1. ...' under the DECOMPOSITION header."""
    if "DECOMPOSITION:" not in text:
        return []
    after = text.split("DECOMPOSITION:", 1)[1]
    before_compose = after.split("COMPOSITION:")[0] if "COMPOSITION:" in after else after
    return re.findall(r"^\s*\d+\.\s*(.+?)\s*$", before_compose, re.MULTILINE)