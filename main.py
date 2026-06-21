"""End-to-end training run using Neural Network agent."""

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
from src.agent.poker_agent import nn_agent_from_weights
from src.training.metrics import profile_agent_strategy, print_strategy_report
from src.cfr.kuhn_cfr import train as cfr_train, CfrAgent, print_cfr_strategy
from src.analysis.plots import plot_learning_curve, plot_strategy_profile


def _print_comparison_table(results: dict) -> None:
    """Drukuje sformatowaną tabelę wyników wszystkich agentów."""
    opponents = list(next(iter(results.values())).keys())
    col_w = 14
    print("\n" + "=" * (16 + col_w * len(opponents)))
    print("   TABELA PORÓWNAWCZA (średni zysk w chipach / rozdanie)")
    print("=" * (16 + col_w * len(opponents)))
    header = f"{'Agent':<16}" + "".join(f"{o:^{col_w}}" for o in opponents)
    print(header)
    print("-" * len(header))
    for agent_name, scores in results.items():
        row = f"{agent_name:<16}" + "".join(f"{scores[o]:^+{col_w}.4f}" for o in opponents)
        print(row)
    print("=" * len(header))


def main() -> None:
    print("=" * 60)
    print("Kuhn Poker — GA + CFR vs Nash Equilibrium")
    print("=" * 60)

    # 1. GA — EWOLUCJA AGENTA SIECI NEURONOWEJ
    best, log = run_evolution(
        weight_size=210,
        agent_from_weights=nn_agent_from_weights,
        pop_size=50,
        generations=100,
        eval_hands=500,
        seed=42,
    )

    print(f"\nNajlepszy fitness treningowy: {best.fitness.values[0]:+.4f}")

    champion = nn_agent_from_weights(best)
    nash_agent = nash_agent_factory(alpha=1 / 6, seed=999)

    # 2. CFR — RÓWNOWAGA NASHA
    print("\n" + "=" * 60)
    print("CFR — wyznaczanie równowagi Nasha")
    print("=" * 60)
    cfr_strategy = cfr_train(iterations=50_000)
    print_cfr_strategy(cfr_strategy)
    cfr_agent = CfrAgent(cfr_strategy)

    # 3. PROFIL STRATEGICZNY GA
    print("\nProfil strategiczny GA champion:")
    ga_profile = profile_agent_strategy(champion)
    print_strategy_report(ga_profile)

    # 4. TABELA PORÓWNAWCZA
    opponents = {
        "vs Expert":    expert_agent,
        "vs AlwaysBet": always_bet_agent,
        "vs Nash":      nash_agent,
        "vs CFR":       cfr_agent,
    }
    results = {}
    for agent_name, agent_obj in [
        ("GA Champion", champion),
        ("CFR Agent",   cfr_agent),
        ("Random",      random_agent),
    ]:
        results[agent_name] = {
            opp_name: play_match(agent_obj, opp_obj, num_hands=2000)
            for opp_name, opp_obj in opponents.items()
        }
    _print_comparison_table(results)

    # 5. ZAPIS I WYKRESY
    with open("best_weights.json", "w") as f:
        json.dump(list(best), f)

    with open("logbook.json", "w") as f:
        json.dump(list(log), f)

    plot_learning_curve("logbook.json", "grafs/learning_curve.png")
    plot_strategy_profile(ga_profile, cfr_strategy, "grafs/strategy_profile.png")

    print("\nZapisano: best_weights.json, logbook.json, grafs/learning_curve.png, grafs/strategy_profile.png")


if __name__ == "__main__":
    main()
