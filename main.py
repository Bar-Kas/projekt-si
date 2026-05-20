"""End-to-end training run.

Wires the class-based ``LinearAgent`` placeholder into the GA, runs the
evolution, and evaluates the champion against the baseline agents.

To switch to Osoba 2's neural network later, change only these lines:

    from src.agent.nn_agent import NeuralAgent
    AGENT_CLASS = NeuralAgent

Everything else stays the same, because a NeuralAgent instance is callable
just like a LinearAgent instance.
"""

from __future__ import annotations

import json

from src.agent.linear_agent import LinearAgent
from src.expert.baseline import (
    always_bet_agent,
    expert_agent,
    nash_agent_factory,
    random_agent,
)
from src.ga.evolution import run_evolution
from src.ga.fitness import play_match

# The agent class the GA will evolve. Swap for NeuralAgent when ready.
AGENT_CLASS = LinearAgent


def main() -> None:
    print("=" * 60)
    print(f"Kuhn Poker - GA training ({AGENT_CLASS.__name__})")
    print("=" * 60)

    best, log = run_evolution(
        weight_size=AGENT_CLASS.WEIGHT_SIZE,
        agent_from_weights=AGENT_CLASS,
        pop_size=60,
        generations=50,
        eval_hands=200,
        seed=42,
    )

    print(f"\nBest training fitness: {best.fitness.values[0]:+.4f}")

    champion = AGENT_CLASS(best)
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

    with open("logbook.json", "w") as f:
        json.dump(
            [
                {k: float(v) for k, v in record.items()
                 if k not in ("gen", "nevals")}
                | {"gen": int(record["gen"]), "nevals": int(record["nevals"])}
                for record in log
            ],
            f,
        )
    print("Saved generation log to logbook.json")


if __name__ == "__main__":
    main()
