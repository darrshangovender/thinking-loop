"""Adjudicator schema / JSON parse tests."""

import pytest
from pydantic import ValidationError

from thinking_loop.adjudicator import Score, AdjudicationResult


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