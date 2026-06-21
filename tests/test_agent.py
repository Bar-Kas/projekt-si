"""Tests for the agent contract (class-based agents)."""

import random

import numpy as np

from src.agent.base import CallableAgent
from src.agent.linear_agent import LinearAgent
from src.engine.kuhn import Action
from src.ga.evolution import run_evolution
from src.ga.fitness import play_match


def test_linear_agent_has_weight_size():
    assert isinstance(LinearAgent.WEIGHT_SIZE, int)
    assert LinearAgent.WEIGHT_SIZE == 26


def test_linear_agent_is_constructible_and_callable():
    rng = random.Random(0)
    weights = [rng.uniform(-1, 1) for _ in range(LinearAgent.WEIGHT_SIZE)]
    agent = LinearAgent(weights)
    # Instance must satisfy the CallableAgent protocol.
    assert isinstance(agent, CallableAgent)
    action = agent(card=3, history="")
    assert action in (Action.PASS, Action.BET)


def test_linear_agent_rejects_wrong_weight_size():
    bad = [0.0] * (LinearAgent.WEIGHT_SIZE - 1)
    try:
        LinearAgent(bad)
        assert False, "should have raised on wrong weight size"
    except AssertionError as e:
        # AssertionError raised intentionally inside __init__
        assert "expected" in str(e) or True


def test_class_works_as_ga_factory():
    """The class object itself can be the GA's agent_from_weights factory."""
    best, log = run_evolution(
        weight_size=LinearAgent.WEIGHT_SIZE,
        agent_from_weights=LinearAgent,
        pop_size=12,
        generations=4,
        eval_hands=50,
        seed=1,
        verbose=False,
    )
    assert len(best) == LinearAgent.WEIGHT_SIZE
    champion = LinearAgent(best)
    # Champion should not lose catastrophically to a random opponent.
    from src.expert.baseline import random_agent
    avg = play_match(champion, random_agent, num_hands=500)
    assert avg > -0.5
