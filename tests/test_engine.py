"""Tests for the engine, encoding, and baseline agents."""

import random

import numpy as np
import pytest

from src.engine.kuhn import (
    Action,
    Card,
    current_player,
    get_payoff,
    is_terminal,
    play_hand,
)
from src.engine.state import STATE_VECTOR_SIZE, encode_state, get_info_set
from src.expert.baseline import (
    always_bet_agent,
    always_pass_agent,
    expert_agent,
    nash_agent_factory,
    random_agent,
)
from src.ga.fitness import play_match


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def test_terminal_histories():
    for h in ("pp", "pbp", "pbb", "bp", "bb"):
        assert is_terminal(h)
    for h in ("", "p", "b", "pb"):
        assert not is_terminal(h)


def test_payoffs_showdown_1():
    assert get_payoff("pp", Card.KING, Card.JACK) == 1
    assert get_payoff("pp", Card.JACK, Card.KING) == -1
    assert get_payoff("pp", Card.QUEEN, Card.JACK) == 1


def test_payoffs_showdown_2():
    assert get_payoff("bb", Card.KING, Card.QUEEN) == 2
    assert get_payoff("pbb", Card.JACK, Card.QUEEN) == -2


def test_payoffs_fold():
    assert get_payoff("bp", Card.JACK, Card.KING) == 1   # P2 folded
    assert get_payoff("pbp", Card.KING, Card.QUEEN) == -1  # P1 folded


def test_current_player_turn_order():
    assert current_player("") == 0
    assert current_player("p") == 1
    assert current_player("b") == 1
    assert current_player("pb") == 0


def test_play_hand_terminates():
    rng = random.Random(7)
    for _ in range(200):
        res = play_hand(random_agent, random_agent, rng)
        assert is_terminal(res.history)
        assert res.p1_payoff == -res.p2_payoff


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def test_encoding_shape_and_card_bit():
    vec = encode_state(Card.KING, "")
    assert vec.shape == (STATE_VECTOR_SIZE,)
    assert vec[Card.KING - 1] == 1.0


def test_encoding_history_positions():
    vec = encode_state(Card.QUEEN, "pb")
    # position 0 = pass
    assert vec[3 + 1] == 1.0
    # position 1 = bet
    assert vec[6 + 2] == 1.0
    # position 2 = empty
    assert vec[9 + 0] == 1.0


def test_info_set_is_unique_per_state():
    assert get_info_set(Card.JACK, "") != get_info_set(Card.JACK, "p")
    assert get_info_set(Card.JACK, "") != get_info_set(Card.QUEEN, "")


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def test_expert_beats_random():
    rng = random.Random(123)
    avg = play_match(expert_agent, random_agent, num_hands=4000, rng=rng)
    assert avg > 0, f"expert should beat random, got {avg}"


def test_always_bet_beats_always_pass():
    rng = random.Random(123)
    avg = play_match(always_bet_agent, always_pass_agent, num_hands=2000, rng=rng)
    assert avg > 0


def test_nash_is_approximately_balanced():
    """A Nash agent vs itself yields close to zero EV."""
    rng = random.Random(123)
    nash_a = nash_agent_factory(alpha=1 / 6, seed=1)
    nash_b = nash_agent_factory(alpha=1 / 6, seed=2)
    avg = play_match(nash_a, nash_b, num_hands=10000, rng=rng)
    assert abs(avg) < 0.05, f"Nash self-play should be near zero, got {avg}"
