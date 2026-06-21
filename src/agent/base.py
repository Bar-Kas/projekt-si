"""Agent contract shared between Osoba 1 (silnik + GA) and Osoba 2 (sieć).

The genetic algorithm produces a flat vector of real-valued weights and
needs to turn it into a playable agent. To keep the two halves of the
project decoupled, we agree on a single contract that ANY agent class
must satisfy:

  1. Class attribute ``WEIGHT_SIZE: int`` -- the length of the genome the
     GA must generate.
  2. Constructor ``__init__(self, weights)`` -- builds the agent from a
     flat weight vector of length ``WEIGHT_SIZE``.
  3. Method ``__call__(self, card: int, history: str) -> Action`` -- makes
     the instance callable, so it is a drop-in replacement for the
     function-based baseline agents used by the engine.

Because instances are callable, ``play_hand`` and ``play_match`` work with
them unchanged. Because the class itself takes ``weights`` in its
constructor, the GA can use the class object directly as its
``agent_from_weights`` factory:

    run_evolution(MyAgent.WEIGHT_SIZE, MyAgent, ...)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.engine.kuhn import Action


@runtime_checkable
class CallableAgent(Protocol):
    """Anything that can decide an action given a game state."""

    def __call__(self, card: int, history: str) -> Action: ...


@runtime_checkable
class WeightAgentFactory(Protocol):
    """A class (or factory) that builds a CallableAgent from a weight vector."""

    WEIGHT_SIZE: int

    def __call__(self, weights) -> CallableAgent: ...


class BaseWeightAgent:
    """Optional convenience base class.

    Subclasses must set ``WEIGHT_SIZE`` and implement ``__init__`` and
    ``__call__``. Provided so Osoba 2 has a clear skeleton to inherit from,
    but inheritance is not required -- duck typing is enough.
    """

    WEIGHT_SIZE: int = 0

    def __init__(self, weights):  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, card: int, history: str) -> Action:  # pragma: no cover
        raise NotImplementedError
