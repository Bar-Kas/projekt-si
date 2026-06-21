"""Class-based linear-policy agent.

This is the reference implementation of the agent contract defined in
``src/agent/base.py``. It plays two roles:

  * **Worked example** -- Osoba 2 can copy this structure for the neural
    network agent. Same class attribute, same constructor signature, same
    ``__call__``.
  * **Drop-in placeholder** -- lets the GA train end-to-end before the
    neural network is finished.

Policy: ``logits = W @ encode_state(card, history) + b``, action = argmax.
"""

from __future__ import annotations

import numpy as np

from src.agent.base import BaseWeightAgent
from src.engine.kuhn import Action
from src.engine.state import NUM_ACTIONS, STATE_VECTOR_SIZE, encode_state


class LinearAgent(BaseWeightAgent):
    """Deterministic linear policy over the encoded game state."""

    # Genome layout: [ W (NUM_ACTIONS x STATE_VECTOR_SIZE) flattened ; b (NUM_ACTIONS) ]
    WEIGHT_SIZE = STATE_VECTOR_SIZE * NUM_ACTIONS + NUM_ACTIONS  # 12 * 2 + 2 = 26

    def __init__(self, weights):
        w = np.asarray(weights, dtype=np.float32)
        assert w.shape == (self.WEIGHT_SIZE,), (
            f"expected {self.WEIGHT_SIZE} weights, got {w.shape}"
        )
        split = STATE_VECTOR_SIZE * NUM_ACTIONS
        self.W = w[:split].reshape(NUM_ACTIONS, STATE_VECTOR_SIZE)
        self.b = w[split:]

    def __call__(self, card: int, history: str) -> Action:
        state = encode_state(card, history)
        logits = self.W @ state + self.b
        return Action(int(np.argmax(logits)))
