"""Confidence calibration.

Combines three signals into a 0-1 confidence score for the winning answer:
  1. Cross-strategy agreement: fraction of strategies whose normalised answer matches the winner
  2. Adjudicator score: normalised winner score / max possible
  3. Adjudicator score variance: high variance = uncertain ranking; low variance = clear winner

Weights are tunable but default to (0.5, 0.3, 0.2). These are NOT theoretically
optimal — they're empirically reasonable for the bench I ship with. Calibrate
on your own benchmark for production use.
"""

from __future__ import annotations

import statistics
from typing import Iterable

from .adjudicator import AdjudicationResult, Score
from .strategies.base import Candidate
from .strategies.self_consistency import _normalise_answer


_AGREEMENT_WEIGHT = 0.5
_SCORE_WEIGHT = 0.3
_VARIANCE_WEIGHT = 0.2


def calibrate(candidates: list[Candidate], adjudication: AdjudicationResult) -> float:
    if not candidates:
        return 0.0
    winner = candidates[adjudication.winner_index]

    # 1. Agreement: how many candidates' normalised answers match the winner's
    winner_norm = _normalise_answer(winner.answer)
    matches = sum(1 for c in candidates if _normalise_answer(c.answer) == winner_norm)
    agreement = matches / len(candidates)

    # 2. Adjudicator score on the winner / max
    winner_score = next((s for s in adjudication.scores if s.candidate_index == adjudication.winner_index), None)
    if winner_score is None:
        score_norm = 0.5
    else:
        score_norm = winner_score.total / 7.0  # max = 3 + 2 + 2

    # 3. Variance of scores — low variance + high winner = confident
    totals = [s.total for s in adjudication.scores]
    if len(totals) >= 2:
        stdev = statistics.pstdev(totals)
        # invert + clip to [0,1]: large stdev → low confidence
        variance_signal = max(0.0, 1.0 - stdev / 3.5)  # 3.5 is half the rubric range
    else:
        variance_signal = 0.5

    raw = _AGREEMENT_WEIGHT * agreement + _SCORE_WEIGHT * score_norm + _VARIANCE_WEIGHT * variance_signal
    return max(0.0, min(1.0, raw))