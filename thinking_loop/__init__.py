"""thinking-loop — test-time reasoning compute with measurable lift.

Public surface:
    ThinkingLoop, Answer       — main orchestrator + result
    Budget                     — token + time + cost cap
    Strategy, Candidate        — base classes for custom strategies
    Trace, TraceEvent          — observability
"""

from .core import ThinkingLoop, Answer
from .budget import Budget, BudgetExceeded
from .trace import Trace, TraceEvent
from .strategies.base import Strategy, Candidate
from .adjudicator import Adjudicator, AdjudicationResult
from .confidence import calibrate

__version__ = "0.1.0"
__all__ = [
    "ThinkingLoop", "Answer",
    "Budget", "BudgetExceeded",
    "Trace", "TraceEvent",
    "Strategy", "Candidate",
    "Adjudicator", "AdjudicationResult",
    "calibrate",
]