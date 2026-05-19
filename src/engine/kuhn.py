"""Kuhn Poker game engine.

Kuhn Poker is a simplified 2-player poker game widely used in
game-theory and AI research.

Rules
-----
* Deck: 3 cards: Jack (1), Queen (2), King (3).
* Each player antes 1 chip into the pot.
* Each player is dealt 1 private card; the third stays in the deck.
* Player 1 acts first. Actions are encoded as:
    'p' = pass  (check if no bet is pending, fold if facing a bet)
    'b' = bet   (bet 1 chip if no bet is pending, call if facing a bet)
* Terminal histories and Player 1 payoffs:
    'pp'  -> showdown for 1 chip (winner's net = +1 or -1)
    'pbp' -> Player 1 folds after Player 2 bets:        -1
    'pbb' -> showdown for 2 chips:                      +/-2
    'bp'  -> Player 2 folds after Player 1 bets:        +1
    'bb'  -> showdown for 2 chips:                      +/-2
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum


class Card(IntEnum):
    JACK = 1
    QUEEN = 2
    KING = 3


class Action(IntEnum):
    PASS = 0
    BET = 1


# All terminal histories. Values that are integers are direct payoffs to
# Player 1. The string sentinels indicate a showdown for that pot size.
TERMINAL_PAYOFFS = {
    "pp": "showdown_1",
    "pbp": -1,
    "pbb": "showdown_2",
    "bp": 1,
    "bb": "showdown_2",
}


def is_terminal(history: str) -> bool:
    """Return True iff the history string is a terminal node of the game."""
    return history in TERMINAL_PAYOFFS


def current_player(history: str) -> int:
    """Player whose turn it is to act. 0 = Player 1, 1 = Player 2."""
    return len(history) % 2


def get_payoff(history: str, p1_card: int, p2_card: int) -> int:
    """Player 1's payoff at a terminal node. Player 2's is the negation."""
    outcome = TERMINAL_PAYOFFS[history]
    if outcome == "showdown_1":
        return 1 if p1_card > p2_card else -1
    if outcome == "showdown_2":
        return 2 if p1_card > p2_card else -2
    return outcome


def legal_actions(history: str) -> list[Action]:
    """In Kuhn Poker every non-terminal node has the same two actions."""
    return [Action.PASS, Action.BET]


@dataclass
class GameResult:
    p1_card: int
    p2_card: int
    history: str
    p1_payoff: int
    p2_payoff: int


def deal_cards(rng: random.Random | None = None) -> tuple[int, int]:
    """Deal the two private cards. Third card stays hidden."""
    if rng is None:
        rng = random
    deck = [Card.JACK, Card.QUEEN, Card.KING]
    rng.shuffle(deck)
    return int(deck[0]), int(deck[1])


def play_hand(agent1, agent2, rng: random.Random | None = None) -> GameResult:
    """Play one full hand. Each agent is a callable: (card, history) -> Action."""
    p1_card, p2_card = deal_cards(rng)
    history = ""

    while not is_terminal(history):
        player = current_player(history)
        card = p1_card if player == 0 else p2_card
        agent = agent1 if player == 0 else agent2
        action = agent(card, history)
        history += "p" if action == Action.PASS else "b"

    p1_payoff = get_payoff(history, p1_card, p2_card)
    return GameResult(p1_card, p2_card, history, p1_payoff, -p1_payoff)
