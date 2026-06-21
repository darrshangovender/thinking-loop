"""Demo: general reasoning question (logical / commonsense)."""

from thinking_loop import Budget, ThinkingLoop
from thinking_loop.strategies import ChainOfThought, Decomposition, Direct, SelfConsistency


def main() -> None:
    loop = ThinkingLoop(
        strategies=[
            Direct(model="claude-haiku-4-5"),
            ChainOfThought(model="claude-sonnet-4-5"),
            Decomposition(model="claude-sonnet-4-5"),
            SelfConsistency(model="claude-sonnet-4-5", n=5, temperature=0.6),
        ],
        adjudicator_model="claude-opus-4-7",
        budget=Budget(max_tokens=15_000, max_seconds=30, max_cost_usd=0.30),
    )

    q = (
        "Alice, Bob, Carol, and Dave stand in a line. "
        "Alice is not first or last. "
        "Bob is directly behind Carol. "
        "Dave is in front of Alice. "
        "Who is at each position?"
    )
    a = loop.solve(q)
    print(f"Final: {a.final}")
    print(f"Confidence: {a.confidence:.2f}  Winner: {a.winning_strategy}")
    print(f"Trace: {a.trace.summary()}")


if __name__ == "__main__":
    main()