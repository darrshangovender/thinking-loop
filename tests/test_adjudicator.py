"""Adjudicator schema / JSON parse tests."""

import pytest
from pydantic import ValidationError

from thinking_loop.adjudicator import AdjudicationResult, Score


def test_score_validates_range():
    with pytest.raises(ValidationError):
        Score(candidate_index=0, correctness=5, reasoning_quality=2, calibration=2)
    with pytest.raises(ValidationError):
        Score(candidate_index=0, correctness=-1, reasoning_quality=2, calibration=2)


def test_score_total_is_sum():
    s = Score(candidate_index=0, correctness=3, reasoning_quality=2, calibration=2)
    assert s.total == 7


def test_adjudication_result_serializable():
    a = AdjudicationResult(
        scores=[Score(candidate_index=0, correctness=2, reasoning_quality=1, calibration=2)],
        winner_index=0,
        rationale="ok",
    )
    assert a.model_dump()["winner_index"] == 0

def _scores(totals: list[tuple[float, float, float]]) -> list[Score]:
    return [
        Score(candidate_index=i, correctness=c, reasoning_quality=r, calibration=k)
        for i, (c, r, k) in enumerate(totals)
    ]


def test_winner_index_out_of_range_is_repaired():
    """winner_index comes straight from model output. An index past the end used
    to raise IndexError in ThinkingLoop.asolve, outside the adjudicator's own
    try/except, taking the whole run down."""
    from thinking_loop.adjudicator import _resolve_winner

    scores = _scores([(3, 2, 2), (1, 1, 1)])
    assert _resolve_winner(scores, n_candidates=2, claimed=7) == 0


def test_negative_winner_index_does_not_silently_pick_from_the_end():
    from thinking_loop.adjudicator import _resolve_winner

    scores = _scores([(3, 2, 2), (0, 0, 0)])
    # -1 would have resolved to candidate 1 — the one the critic scored zero.
    assert _resolve_winner(scores, n_candidates=2, claimed=-1) == 0


def test_winner_contradicting_its_own_scores_is_overruled():
    """The rubric is authoritative: 'highest score wins'. A critic naming the
    candidate it scored 3/7 over the one it scored 7/7 returned the worse answer."""
    from thinking_loop.adjudicator import _resolve_winner

    scores = _scores([(3, 2, 2), (1, 1, 1)])
    assert _resolve_winner(scores, n_candidates=2, claimed=1) == 0


def test_consistent_winner_is_honoured_unchanged():
    from thinking_loop.adjudicator import _resolve_winner

    scores = _scores([(1, 1, 1), (3, 2, 2)])
    assert _resolve_winner(scores, n_candidates=2, claimed=1) == 1


def test_tie_is_broken_by_reasoning_quality():
    from thinking_loop.adjudicator import _resolve_winner

    # Both total 5; candidate 1 has the higher reasoning_quality.
    scores = _scores([(3, 1, 1), (2, 2, 1)])
    assert _resolve_winner(scores, n_candidates=2, claimed=0) == 1
