"""Fitness evaluation.

The genetic algorithm needs a numeric score per individual. We obtain
the score by playing many hands of Kuhn Poker between the candidate
agent and a pool of fixed opponents. Position is alternated each hand
to remove first-mover bias.
"""

from __future__ import annotations

import random
from typing import Callable, Iterable

from src.engine.kuhn import play_hand

Agent = Callable[[int, str], int]  # (card, history) -> Action


def play_match(
    agent_a: Agent,
    agent_b: Agent,
    num_hands: int = 1000,
    rng: random.Random | None = None,
) -> float:
    """Average chips per hand for agent_a across num_hands.

    Position is alternated each hand to eliminate positional bias.
    """
    if rng is None:
        rng = random.Random()

    total = 0
    for i in range(num_hands):
        if i % 2 == 0:
            res = play_hand(agent_a, agent_b, rng)
            total += res.p1_payoff
        else:
            res = play_hand(agent_b, agent_a, rng)
            total += res.p2_payoff

    return total / num_hands


def evaluate_against_pool(
    candidate: Agent,
    opponents: Iterable[Agent],
    num_hands_each: int = 200,
    rng: random.Random | None = None,
) -> float:
    """Mean per-hand profit of `candidate` over each opponent in the pool."""
    if rng is None:
        rng = random.Random()

    scores = [play_match(candidate, opp, num_hands_each, rng) for opp in opponents]
    return sum(scores) / len(scores)
