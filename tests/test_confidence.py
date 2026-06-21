"""Tests for the confidence calibrator."""

from thinking_loop import calibrate
from thinking_loop.adjudicator import AdjudicationResult, Score
from thinking_loop.strategies.base import Candidate


def _adj(winner: int, scores: list[tuple[float, float, float]]) -> AdjudicationResult:
    return AdjudicationResult(
        scores=[Score(candidate_index=i, correctness=c, reasoning_quality=r, calibration=k) for i, (c, r, k) in enumerate(scores)],
        winner_index=winner,
        rationale="test",
    )


def test_full_agreement_high_score_high_confidence():
    cs = [Candidate(strategy="a", answer="42"), Candidate(strategy="b", answer="42"), Candidate(strategy="c", answer="42")]
    adj = _adj(winner=0, scores=[(3, 2, 2), (3, 2, 2), (3, 2, 2)])
    conf = calibrate(cs, adj)
    assert conf > 0.9


def test_disagreement_lowers_confidence():
    cs = [Candidate(strategy="a", answer="42"), Candidate(strategy="b", answer="100"), Candidate(strategy="c", answer="0")]
    adj = _adj(winner=0, scores=[(3, 2, 2), (1, 1, 1), (0, 0, 0)])
    conf = calibrate(cs, adj)
    assert conf < 0.6


def test_zero_candidates_returns_zero():
    adj = _adj(winner=0, scores=[(0, 0, 0)])
    assert calibrate([], adj) == 0.0