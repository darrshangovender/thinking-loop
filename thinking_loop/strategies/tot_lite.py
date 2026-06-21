"""ToT-Lite — Tree-of-Thoughts (Yao et al., 2023), beam-search variant.

Full ToT is expensive (BFS/DFS over thought tree with a critic at each node).
ToT-Lite is a cheaper approximation:
  - Generate `beam_width` candidate next-thoughts at depth 1
  - Score each with the SAME LLM (acting as a step evaluator: "is this step
    on the right track?")
  - Keep top-K, expand each to depth 2, score, keep top-K
  - At max depth, take the highest-scoring full chain and ask the model for
    the final answer based on that chain

In practice this beats CoT on multi-step reasoning by 3-8 points on hard
benchmarks at 4-8× the cost. Empirically the diminishing returns hit at
depth > 4 and beam_width > 4, so we cap defaults at 3.
"""

from __future__ import annotations

import asyncio
import re

from ..budget import Budget
from ..trace import Trace
from .base import Candidate, Strategy


EXPAND_SYSTEM = """\
You are working step by step on a hard problem. Given the question and the
reasoning so far, propose ONE next reasoning step. Be concrete: state what
you're computing or concluding, not what you intend to compute.

Output just the step, nothing else.
"""

SCORE_SYSTEM = """\
You are evaluating a single reasoning step on a hard problem. Given the
question, the prior reasoning, and the proposed next step, rate the step
on a 0.0-1.0 scale for whether it advances the solution correctly.

Output ONLY the float, e.g. 0.85
"""

FINAL_SYSTEM = """\
You are given a question and a chain of reasoning steps that solved it.
Output the final answer on a single line prefixed with 'Answer: '.
"""


class ToTLite(Strategy):
    name = "tot_lite"

    def __init__(self, *, model: str, beam_width: int = 3, depth: int = 3):
        super().__init__(model=model)
        if beam_width < 1 or depth < 1:
            raise ValueError("beam_width and depth must be >= 1")
        self.beam_width = beam_width
        self.depth = depth

    async def run(self, question: str, *, trace: Trace, budget: Budget) -> Candidate:
        trace.emit("strategy_start", self.name, beam_width=self.beam_width, depth=self.depth)

        # Start with a single empty chain. At each depth, expand each chain in the
        # beam to `beam_width` candidates, score them all, keep top-K.
        beams: list[list[str]] = [[]]
        for d in range(self.depth):
            budget.check()
            expanded: list[list[str]] = []
            # Expand: each chain → beam_width candidate next steps (parallel)
            expand_tasks = [self._propose_step(question, chain, trace, budget) for chain in beams for _ in range(self.beam_width)]
            new_steps = await asyncio.gather(*expand_tasks, return_exceptions=True)
            i = 0
            for chain in beams:
                for _ in range(self.beam_width):
                    s = new_steps[i]
                    i += 1
                    if isinstance(s, Exception) or not s:
                        continue
                    expanded.append(chain + [s])
            if not expanded:
                break
            # Score each expanded chain on its newest step
            score_tasks = [self._score_step(question, chain[:-1], chain[-1], trace, budget) for chain in expanded]
            scores = await asyncio.gather(*score_tasks, return_exceptions=True)
            scored = [(chain, s if not isinstance(s, Exception) else 0.0) for chain, s in zip(expanded, scores)]
            scored.sort(key=lambda x: x[1], reverse=True)
            beams = [c for c, _ in scored[: self.beam_width]]

        # Pick best chain, generate final answer.
        best_chain = beams[0] if beams else []
        chain_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(best_chain))
        text = await self._llm(
            messages=[{"role": "user", "content": f"Question: {question}\n\nReasoning:\n{chain_text}"}],
            system=FINAL_SYSTEM, trace=trace, budget=budget,
            max_tokens=256, temperature=0.2,
        )
        answer = _extract_answer(text)
        trace.emit("strategy_end", self.name, answer=answer, chain_length=len(best_chain))
        return Candidate(
            strategy=self.name, answer=answer,
            reasoning=f"Beam-best chain:\n{chain_text}\n\nFinal: {answer}",
            intermediate={"best_chain": best_chain, "chain_length": len(best_chain)},
        )

    async def _propose_step(self, question: str, chain: list[str], trace: Trace, budget: Budget) -> str:
        prior = "\n".join(f"{i+1}. {s}" for i, s in enumerate(chain)) if chain else "(none yet)"
        return (await self._llm(
            messages=[{"role": "user", "content": f"Question: {question}\n\nReasoning so far:\n{prior}\n\nPropose the next step."}],
            system=EXPAND_SYSTEM, trace=trace, budget=budget,
            max_tokens=200, temperature=0.7,
        )).strip()

    async def _score_step(self, question: str, prior: list[str], step: str, trace: Trace, budget: Budget) -> float:
        prior_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(prior)) if prior else "(none)"
        raw = await self._llm(
            messages=[{"role": "user", "content": f"Question: {question}\n\nPrior reasoning:\n{prior_text}\n\nProposed next step:\n{step}\n\nRate 0.0-1.0:"}],
            system=SCORE_SYSTEM, trace=trace, budget=budget,
            max_tokens=10, temperature=0.0,
        )
        m = re.search(r"\d+(?:\.\d+)?", raw)
        return min(1.0, max(0.0, float(m.group(0)))) if m else 0.0


def _extract_answer(text: str) -> str:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.lower().startswith("answer:"):
            return line.split(":", 1)[1].strip()
    return text.strip().splitlines()[-1] if text.strip() else ""