"""Demo: solve a hard math word problem with the full loop."""

from thinking_loop import Budget, ThinkingLoop
from thinking_loop.strategies import ChainOfThought, Decomposition, Direct, SelfConsistency, ToTLite


def main() -> None:
    loop = ThinkingLoop(
        strategies=[
            Direct(model="claude-haiku-4-5"),
            ChainOfThought(model="claude-sonnet-4-5"),
            SelfConsistency(model="claude-sonnet-4-5", n=5, temperature=0.7),
            Decomposition(model="claude-sonnet-4-5"),
            ToTLite(model="claude-sonnet-4-5", beam_width=3, depth=3),
        ],
        adjudicator_model="claude-opus-4-7",
        budget=Budget(max_tokens=20_000, max_seconds=45, max_cost_usd=0.50),
    )

    question = (
        "If a train leaves Durban at 14:00 doing 80 km/h, and another leaves "
        "Johannesburg at 14:30 doing 110 km/h heading toward Durban (the cities "
        "are 570 km apart), at what time do they meet?"
    )

    answer = loop.solve(question)
    print("Final:           ", answer.final)
    print("Confidence:      ", f"{answer.confidence:.2f}")
    print("Winning strategy:", answer.winning_strategy)
    print("Trace summary:   ", answer.trace.summary())
    print()
    print("All candidates:")
    for c in answer.candidates:
        print(f"  [{c.strategy:20}] {c.answer}")


if __name__ == "__main__":
    main()