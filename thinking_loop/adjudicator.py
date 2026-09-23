"""Adjudicator — critic LLM that scores Candidates and picks a winner.

Rubric (deliberately small + strict):
  - correctness (0-3)         : does the answer look right given the question
  - reasoning_quality (0-2)   : is the chain coherent, free of jumps and hallucinations
  - calibration (0-2)         : does the answer's specificity match what the question demands

Total /7. Highest score wins; ties broken by reasoning_quality.

The rubric — not the model's own `winner_index` — is authoritative. The critic is
asked to report a winner, but it is free to name an index that contradicts its
scores or does not exist; `_resolve_winner` reconciles the two.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError

from .budget import Budget
from .llm import LLM
from .strategies.base import Candidate
from .trace import Trace


class Score(BaseModel):
    candidate_index: int
    correctness: float = Field(ge=0, le=3)
    reasoning_quality: float = Field(ge=0, le=2)
    calibration: float = Field(ge=0, le=2)
    reasoning: str = ""

    @property
    def total(self) -> float:
        return self.correctness + self.reasoning_quality + self.calibration


class AdjudicationResult(BaseModel):
    scores: list[Score]
    winner_index: int
    rationale: str = ""


SYSTEM = """\
You are an adjudicator scoring candidate answers to a hard question.

For each candidate, score on this strict rubric:
  - correctness:       0-3 (3 = clearly correct, 2 = mostly, 1 = partial, 0 = wrong)
  - reasoning_quality: 0-2 (2 = coherent and free of jumps, 1 = mixed, 0 = bad)
  - calibration:       0-2 (2 = specificity matches what the question demands, 0 = off)

Output strict JSON:
{
  "scores": [
    {"candidate_index": 0, "correctness": ..., "reasoning_quality": ..., "calibration": ..., "reasoning": "<one short sentence>"},
    ...
  ],
  "winner_index": <int>,
  "rationale": "<one sentence on why the winner is best>"
}

Tie-break rule: higher reasoning_quality wins. The winner_index is the candidate with the highest total.
"""


class Adjudicator:
    def __init__(self, *, model: str):
        self.model = model
        self.llm = LLM(model)

    async def judge(self, question: str, candidates: list[Candidate], *, trace: Trace, budget: Budget) -> AdjudicationResult:
        trace.emit("adjudicator_call", "adjudicator", n_candidates=len(candidates))
        candidates_block = "\n\n".join(
            f"CANDIDATE {i} (strategy={c.strategy}):\nReasoning: {c.reasoning[:1500]}\nAnswer: {c.answer}"
            for i, c in enumerate(candidates)
        )
        user = f"Question: {question}\n\n{candidates_block}"
        budget.check()
        resp = await self.llm.chat(
            messages=[{"role": "user", "content": user}],
            system=SYSTEM, max_tokens=2048, temperature=0.0,
        )
        trace.emit_llm("adjudicator", model=resp.model, tokens_in=resp.tokens_in, tokens_out=resp.tokens_out, cost_usd=resp.cost_usd, duration_ms=resp.duration_ms)
        budget.add(tokens_in=resp.tokens_in, tokens_out=resp.tokens_out, cost_usd=resp.cost_usd)
        data = _parse_json(resp.content)
        try:
            result = AdjudicationResult(**data)
        except ValidationError as e:
            raise ValueError(f"adjudicator returned invalid JSON: {e}") from e
        result.winner_index = _resolve_winner(result.scores, len(candidates), result.winner_index)
        return result


def _resolve_winner(scores: list[Score], n_candidates: int, claimed: int) -> int:
    """Reconcile the adjudicator's claimed winner against its own scores.

    ``winner_index`` arrives straight from model output, so it can (a) point past
    the end of the candidate list — an IndexError one frame later, after the
    adjudicator's try/except has already been left — or (b) be negative, which
    Python resolves silently to a candidate counting from the end. Worse, it can
    simply disagree with the rubric and hand back the answer the critic itself
    scored lowest.

    The rubric decides: highest total, ties broken by reasoning_quality. A claim
    that is already top-ranked is honoured as-is, so a well-behaved adjudicator
    sees no change in behaviour.
    """
    usable = [s for s in scores if 0 <= s.candidate_index < n_candidates]
    if not usable:
        return claimed if 0 <= claimed < n_candidates else 0
    rank = lambda s: (s.total, s.reasoning_quality)  # noqa: E731
    best = max(usable, key=rank)
    claimed_score = next((s for s in usable if s.candidate_index == claimed), None)
    if claimed_score is not None and rank(claimed_score) == rank(best):
        return claimed
    return best.candidate_index


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"no JSON in adjudicator output: {text[:200]!r}")
    return json.loads(m.group(0))