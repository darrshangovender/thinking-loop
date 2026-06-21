"""Reproducible benchmark for the strategies + the full loop.

Run:
    cd benchmarks
    python run.py --model claude-sonnet-4-5 --adjudicator claude-opus-4-7

Results are written to benchmarks/results.json. The README's benchmark
table is generated from these results.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import yaml

from thinking_loop import Budget, ThinkingLoop
from thinking_loop.strategies import ChainOfThought, Decomposition, Direct, SelfConsistency, ToTLite
from thinking_loop.strategies.self_consistency import _normalise_answer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-sonnet-4-5")
    parser.add_argument("--adjudicator", default="claude-opus-4-7")
    parser.add_argument("--budget-tokens", type=int, default=30_000)
    parser.add_argument("--budget-seconds", type=float, default=60.0)
    parser.add_argument("--budget-cost", type=float, default=1.00)
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()

    bench_file = Path(__file__).parent / "gsm8k_subset.yml"
    questions = yaml.safe_load(bench_file.read_text())
    print(f"Loaded {len(questions)} benchmark questions")

    loop = ThinkingLoop(
        strategies=[
            Direct(model=args.model),
            ChainOfThought(model=args.model),
            Decomposition(model=args.model),
            SelfConsistency(model=args.model, n=5, temperature=0.7),
            ToTLite(model=args.model, beam_width=3, depth=3),
        ],
        adjudicator_model=args.adjudicator,
        budget=Budget(max_tokens=args.budget_tokens, max_seconds=args.budget_seconds, max_cost_usd=args.budget_cost),
    )

    per_strategy: dict[str, dict] = {}
    loop_correct = 0
    loop_latencies: list[float] = []
    loop_costs: list[float] = []

    for q in questions:
        print(f"  Q {q['id']:8} ", end="", flush=True)
        t0 = time.perf_counter()
        ans = loop.solve(q["question"])
        elapsed = time.perf_counter() - t0
        gold = _normalise_answer(q["answer"])

        # Per-strategy bookkeeping
        for c in ans.candidates:
            s = per_strategy.setdefault(c.strategy, {"correct": 0, "total": 0, "latencies": [], "costs": []})
            s["total"] += 1
            if _normalise_answer(c.answer) == gold:
                s["correct"] += 1
            # Strategy-specific cost: sum LLM events for that actor
            actor_events = [e for e in ans.trace.events if e.actor == c.strategy and e.kind == "llm_call"]
            s["latencies"].append(sum(e.duration_ms for e in actor_events) / 1000.0)
            s["costs"].append(sum(e.cost_usd or 0 for e in actor_events))

        # Loop bookkeeping
        is_correct = _normalise_answer(ans.final) == gold
        if is_correct:
            loop_correct += 1
        loop_latencies.append(elapsed)
        loop_costs.append(ans.trace.total_cost_usd())
        print(f" loop:{'✓' if is_correct else '✗'} ({ans.winning_strategy}, conf={ans.confidence:.2f})")

    # Aggregate
    rows = []
    for name, s in per_strategy.items():
        rows.append({
            "strategy": name,
            "accuracy": s["correct"] / s["total"] if s["total"] else 0,
            "median_latency_s": statistics.median(s["latencies"]) if s["latencies"] else 0,
            "median_cost_usd": statistics.median(s["costs"]) if s["costs"] else 0,
        })
    rows.append({
        "strategy": "ThinkingLoop",
        "accuracy": loop_correct / len(questions),
        "median_latency_s": statistics.median(loop_latencies),
        "median_cost_usd": statistics.median(loop_costs),
    })

    out_path = Path(__file__).parent / args.out
    out_path.write_text(json.dumps({"model": args.model, "adjudicator": args.adjudicator, "results": rows, "n_questions": len(questions)}, indent=2))
    print(f"\nWrote {out_path}")
    print("\nSummary:")
    for r in rows:
        print(f"  {r['strategy']:20} acc={r['accuracy']:.1%}  lat={r['median_latency_s']:.1f}s  cost=${r['median_cost_usd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())