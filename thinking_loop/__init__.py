"""thinking-loop — test-time reasoning compute with measurable lift.

Public surface:
    ThinkingLoop, Answer       — main orchestrator + result
    Budget                     — token + time + cost cap
    Strategy, Candidate        — base classes for custom strategies
    Trace, TraceEvent          — observability
"""

from .adjudicator import AdjudicationResult, Adjudicator
from .budget import Budget, BudgetExceeded
from .confidence import calibrate
from .core import Answer, ThinkingLoop
from .strategies.base import Candidate, Strategy
from .trace import Trace, TraceEvent

__version__ = "0.1.0"
__all__ = [
    "AdjudicationResult",
    "Adjudicator",
    "Answer",
    "Budget",
    "BudgetExceeded",
    "Candidate",
    "Strategy",
    "ThinkingLoop",
    "Trace",
    "TraceEvent",
    "calibrate",
]