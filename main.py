"""End-to-end training run.

This script wires the linear-policy placeholder into the GA, runs the
evolution for a few generations, and then evaluates the champion against
the baseline agents. Person 2 will swap `linear_agent_from_weights` for
a neural-network factory once the NN module is ready.
"""

from __future__ import annotations

import json

from src.expert.baseline import (
    always_bet_agent,
    expert_agent,
    nash_agent_factory,
    random_agent,
)
from src.ga.evolution import run_evolution
from src.ga.fitness import play_match
from src.ga.linear_agent import WEIGHT_SIZE, linear_agent_from_weights


def main() -> None:
    print("=" * 60)
    print("Kuhn Poker - GA training (linear-policy placeholder)")
    print("=" * 60)

    best, log = run_evolution(
        weight_size=WEIGHT_SIZE,
        agent_from_weights=linear_agent_from_weights,
        pop_size=60,
        generations=50,
        eval_hands=200,
        seed=42,
    )

    print(f"\nBest training fitness: {best.fitness.values[0]:+.4f}")

    champion = linear_agent_from_weights(best)
    nash = nash_agent_factory(alpha=1 / 6, seed=999)

    print("\nFinal evaluation (2000 hands each, position alternated):")
    for name, opp in [
        ("expert", expert_agent),
        ("random", random_agent),
        ("always_bet", always_bet_agent),
        ("nash", nash),
    ]:
        avg = play_match(champion, opp, num_hands=2000)
        print(f"  vs {name:<10s}: {avg:+.4f} chips/hand")

    with open("best_weights.json", "w") as f:
        json.dump(list(best), f)
    print("\nSaved champion weights to best_weights.json")

    # Save the per-generation logbook for plotting in the report.
    with open("logbook.json", "w") as f:
        json.dump(
            [{k: float(v) for k, v in record.items() if k != "gen" and k != "nevals"}
             | {"gen": int(record["gen"]), "nevals": int(record["nevals"])}
             for record in log],
            f,
        )
    print("Saved generation log to logbook.json")


if __name__ == "__main__":
    main()
