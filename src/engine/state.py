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


# ---------------------------------------------------------------------------
# Adapter for Osoba 2's dictionary format (Grzegorz)
# ---------------------------------------------------------------------------

CARD_INT_TO_STR = {1: "J", 2: "Q", 3: "K"}
CARD_STR_TO_INT = {"J": 1, "Q": 2, "K": 3}
_ACTION_CHAR_TO_WORD = {"p": "check", "b": "bet"}
_ACTION_WORD_TO_CHAR = {"check": "p", "bet": "b", "fold": "p", "call": "b"}


def build_state_dict(card: int, history: str) -> dict:
    """Build the dictionary format expected by Osoba 2's encoder.

    Returns e.g. {"my_card": "K", "position": 0, "history": ["check", "bet"]}.
    """
    return {
        "my_card": CARD_INT_TO_STR[card],
        "position": len(history) % 2,  # 0 = Player 1, 1 = Player 2
        "history": [_ACTION_CHAR_TO_WORD[a] for a in history],
    }


def encode_from_dict(state_dict: dict) -> np.ndarray:
    """Encode Osoba 2's dictionary format into the 12-dim feature vector."""
    card = CARD_STR_TO_INT[state_dict["my_card"]]
    history = "".join(_ACTION_WORD_TO_CHAR[a] for a in state_dict["history"])
    return encode_state(card, history)
