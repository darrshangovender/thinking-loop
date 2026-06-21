"""Budget guard tests."""

import time

import pytest

from thinking_loop import Budget, BudgetExceeded


def test_token_cap_enforced():
    b = Budget(max_tokens=100, max_seconds=10, max_cost_usd=1.0)
    b.add(tokens_in=50, tokens_out=60, cost_usd=0.001)
    with pytest.raises(BudgetExceeded):
        b.check()


def test_cost_cap_enforced():
    b = Budget(max_tokens=1_000_000, max_seconds=10, max_cost_usd=0.01)
    b.add(tokens_in=10, tokens_out=10, cost_usd=0.02)
    with pytest.raises(BudgetExceeded):
        b.check()


def test_time_cap_enforced():
    b = Budget(max_tokens=1_000_000, max_seconds=0.01, max_cost_usd=1.0)
    time.sleep(0.05)
    with pytest.raises(BudgetExceeded):
        b.check()


def test_reset_clears_counters():
    b = Budget()
    b.add(tokens_in=100, tokens_out=200, cost_usd=0.5)
    b.reset()
    assert b.total_tokens == 0
    assert b.cost_usd == 0.0