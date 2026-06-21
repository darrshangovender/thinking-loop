"""Tests for strategies that don't require real LLM calls."""

import pytest

from thinking_loop import Budget
from thinking_loop.strategies import Direct, ChainOfThought, SelfConsistency, Decomposition, ToTLite
from thinking_loop.strategies.self_consistency import _normalise_answer
from thinking_loop.strategies.cot import _extract_answer as cot_extract
from thinking_loop.strategies.decomposition import _extract_sub_questions


def test_self_consistency_normalises_numbers():
    assert _normalise_answer("5") == "5"
    assert _normalise_answer("5.00") == "5"
    assert _normalise_answer("about 5") == "5"
    assert _normalise_answer("The answer is 5.") == "5"


def test_self_consistency_normalises_text():
    assert _normalise_answer("Hello!") == "hello"
    assert _normalise_answer("  Yes.  ") == "yes"


def test_self_consistency_validates_n():
    with pytest.raises(ValueError):
        SelfConsistency(model="claude-haiku-4-5", n=0)


def test_tot_lite_validates_args():
    with pytest.raises(ValueError):
        ToTLite(model="claude-haiku-4-5", beam_width=0, depth=1)
    with pytest.raises(ValueError):
        ToTLite(model="claude-haiku-4-5", beam_width=1, depth=0)


def test_cot_extracts_answer_format():
    text = "THOUGHT: stuff\n1. one\n2. two\nAnswer: 42"
    assert cot_extract(text) == "42"


def test_decomposition_extracts_sub_questions():
    text = """DECOMPOSITION:
1. What is 2+2?
   It's 4.
2. What is 4+1?
   It's 5.

COMPOSITION: combine them
Answer: 5"""
    subs = _extract_sub_questions(text)
    assert len(subs) == 2
    assert "What is 2+2?" in subs