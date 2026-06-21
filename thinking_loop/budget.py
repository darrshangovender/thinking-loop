"""Hard budget guards for the loop.

Three independent caps: max tokens, max wall-clock seconds, max USD cost.
Whichever is hit first raises BudgetExceeded. Strategies that hold the
guard check at each LLM call get cancelled cleanly rather than running
to completion past the cap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    pass


@dataclass
class Budget:
    """Per-call budget. Mutable counters are updated as the call runs."""
    max_tokens: int = 50_000
    max_seconds: float = 60.0
    max_cost_usd: float = 1.00

    # Mutable running totals
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    _started_at: float = field(default_factory=time.perf_counter)

    def reset(self) -> None:
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost_usd = 0.0
        self._started_at = time.perf_counter()

    def add(self, *, tokens_in: int, tokens_out: int, cost_usd: float | None) -> None:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        if cost_usd is not None:
            self.cost_usd += cost_usd

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._started_at

    def check(self) -> None:
        """Raise BudgetExceeded if any cap is breached. Strategies call this between LLM calls."""
        if self.total_tokens >= self.max_tokens:
            raise BudgetExceeded(f"tokens: {self.total_tokens} >= {self.max_tokens}")
        if self.elapsed >= self.max_seconds:
            raise BudgetExceeded(f"seconds: {self.elapsed:.1f} >= {self.max_seconds}")
        if self.cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(f"cost: ${self.cost_usd:.4f} >= ${self.max_cost_usd:.4f}")

    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed)