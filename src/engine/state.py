"""State representation and encoding for Kuhn Poker.

The encoding converts (card, history) into a fixed-size feature vector
that any agent (rule-based, linear, neural network) can consume.

Encoding layout (12 dimensions total):
  - dims [0:3]  : one-hot of the player's card (J, Q, K)
  - dims [3:6]  : history position 0:  [empty, pass, bet]
  - dims [6:9]  : history position 1:  [empty, pass, bet]
  - dims [9:12] : history position 2:  [empty, pass, bet]

Kuhn Poker has at most 3 actions before a terminal state, so 3 history
slots is sufficient.
"""

from __future__ import annotations

import numpy as np

STATE_VECTOR_SIZE = 12
NUM_ACTIONS = 2
MAX_HISTORY_LEN = 3


def get_info_set(card: int, history: str) -> str:
    """Human-readable information set key used for logging and analysis."""
    return f"{card}:{history}"


def encode_state(card: int, history: str) -> np.ndarray:
    """Encode (card, history) as a float32 vector of length STATE_VECTOR_SIZE."""
    vec = np.zeros(STATE_VECTOR_SIZE, dtype=np.float32)

    # Card one-hot.
    vec[card - 1] = 1.0

    # History positions.
    for i in range(MAX_HISTORY_LEN):
        offset = 3 + i * 3
        if i < len(history):
            ch = history[i]
            if ch == "p":
                vec[offset + 1] = 1.0
            elif ch == "b":
                vec[offset + 2] = 1.0
        else:
            # No action yet at this position.
            vec[offset] = 1.0

    return vec
