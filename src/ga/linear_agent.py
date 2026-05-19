"""Placeholder linear-policy agent.

The GA evolves a flat vector of real-valued weights. Person 2 will later
supply a `agent_from_weights` factory that builds a neural network from
the vector. Until then, this simple linear policy lets us test the GA
end-to-end and serves as a baseline:

    logits = W @ encode_state(card, history) + b
    action = argmax(logits)
"""

from __future__ import annotations

import numpy as np

from src.engine.kuhn import Action
from src.engine.state import NUM_ACTIONS, STATE_VECTOR_SIZE, encode_state

# Weight vector layout: [W_flat ; b]
WEIGHT_SIZE = STATE_VECTOR_SIZE * NUM_ACTIONS + NUM_ACTIONS  # 12 * 2 + 2 = 26


def linear_agent_from_weights(weights):
    """Build a deterministic agent from a flat weight vector of length WEIGHT_SIZE."""
    w = np.asarray(weights, dtype=np.float32)
    assert w.shape == (WEIGHT_SIZE,), f"expected {WEIGHT_SIZE} weights, got {w.shape}"

    W = w[: STATE_VECTOR_SIZE * NUM_ACTIONS].reshape(NUM_ACTIONS, STATE_VECTOR_SIZE)
    b = w[STATE_VECTOR_SIZE * NUM_ACTIONS :]

    def agent(card: int, history: str) -> Action:
        state = encode_state(card, history)
        logits = W @ state + b
        return Action(int(np.argmax(logits)))

    return agent
