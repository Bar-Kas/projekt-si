"""Baseline agents for Kuhn Poker.

These provide opponents for evolutionary training and benchmarks for
the report. Each agent is a callable with signature (card, history) -> Action.
"""

from __future__ import annotations

import random

from src.engine.kuhn import Action, Card


def random_agent(card: int, history: str, rng: random.Random | None = None) -> Action:
    """Pick PASS or BET uniformly at random."""
    if rng is None:
        rng = random
    return rng.choice([Action.PASS, Action.BET])


def always_pass_agent(card: int, history: str) -> Action:
    """Always pass. Loses heavily; useful as a sanity-check opponent."""
    return Action.PASS


def always_bet_agent(card: int, history: str) -> Action:
    """Always bet. Very aggressive; useful as a sanity-check opponent."""
    return Action.BET


def expert_agent(card: int, history: str) -> Action:
    """Hand-crafted rule-based expert ('system ekspertowy').

    Encodes intuitive poker heuristics:
      - King  : always value-bet / call.
      - Queen : check-call (cautious medium hand).
      - Jack  : check; fold to a bet (weak hand).
    """
    if card == Card.KING:
        return Action.BET

    if card == Card.QUEEN:
        if history in ("b", "pb"):
            return Action.BET  # call the bet
        return Action.PASS     # otherwise check

    # Card.JACK
    return Action.PASS         # never bet, fold to any bet


def nash_agent_factory(alpha: float = 1 / 6, seed: int | None = None):
    """Build an agent that plays a Nash-equilibrium strategy.

    Kuhn Poker has a 1-parameter family of equilibria with alpha in [0, 1/3].
    The default alpha = 1/6 is the midpoint of the family.

    Returns a callable: (card, history) -> Action.
    """
    rng = random.Random(seed)

    def agent(card: int, history: str) -> Action:
        # ----- Player 1 (acts on '' and 'pb') -----
        if history == "":
            if card == Card.JACK:
                return Action.BET if rng.random() < alpha else Action.PASS
            if card == Card.QUEEN:
                return Action.PASS
            return Action.BET if rng.random() < 3 * alpha else Action.PASS

        if history == "pb":
            if card == Card.JACK:
                return Action.PASS  # fold
            if card == Card.QUEEN:
                return Action.BET if rng.random() < alpha + 1 / 3 else Action.PASS
            return Action.BET       # King always calls

        # ----- Player 2 (acts on 'p' and 'b') -----
        if history == "p":
            if card == Card.JACK:
                return Action.BET if rng.random() < 1 / 3 else Action.PASS
            if card == Card.QUEEN:
                return Action.PASS
            return Action.BET

        if history == "b":
            if card == Card.JACK:
                return Action.PASS  # fold
            if card == Card.QUEEN:
                return Action.BET if rng.random() < 1 / 3 else Action.PASS
            return Action.BET       # King calls

        return Action.PASS  # unreachable

    return agent
