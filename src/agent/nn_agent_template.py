"""SZABLON dla Osoby 2 (Grzegorz) -- agent oparty na sieci neuronowej.

Wypełnij miejsca oznaczone TODO. Zachowaj kontrakt:
  - klasowy atrybut WEIGHT_SIZE,
  - konstruktor __init__(self, weights),
  - metoda __call__(self, card, history) -> Action.

Dzięki temu klasa jest gotowa do użycia w GA bez żadnych zmian po
stronie Osoby 1:

    from src.agent.nn_agent import NeuralAgent
    best, log = run_evolution(NeuralAgent.WEIGHT_SIZE, NeuralAgent, ...)

UWAGA: ten plik jest szkieletem -- domyślnie podnosi NotImplementedError.
Usuń wyjątki, gdy zaimplementujesz sieć.
"""

from __future__ import annotations

import numpy as np

# import torch  # odkomentuj gdy zaczniesz implementować

from src.agent.base import BaseWeightAgent
from src.engine.kuhn import Action
from src.engine.state import NUM_ACTIONS, STATE_VECTOR_SIZE, encode_state


# --- Definicja architektury (przykład MLP 12 -> 16 -> 2) -------------------
# Ustal rozmiary warstw i policz liczbę wszystkich wag + biasów. Ta liczba
# MUSI być równa WEIGHT_SIZE, bo tyle genów wygeneruje GA.
HIDDEN_SIZE = 16

# Liczba parametrów MLP: (in*hidden + hidden) + (hidden*out + out)
_W1 = STATE_VECTOR_SIZE * HIDDEN_SIZE          # wagi warstwy 1
_B1 = HIDDEN_SIZE                              # biasy warstwy 1
_W2 = HIDDEN_SIZE * NUM_ACTIONS                # wagi warstwy 2
_B2 = NUM_ACTIONS                              # biasy warstwy 2
_TOTAL = _W1 + _B1 + _W2 + _B2                 # = 12*16 + 16 + 16*2 + 2 = 242


class NeuralAgent(BaseWeightAgent):
    """Agent decyzyjny oparty na MLP. Wagi pochodzą z algorytmu genetycznego."""

    WEIGHT_SIZE = _TOTAL  # 242 dla domyślnej architektury powyżej

    def __init__(self, weights):
        w = np.asarray(weights, dtype=np.float32)
        assert w.shape == (self.WEIGHT_SIZE,), (
            f"expected {self.WEIGHT_SIZE} weights, got {w.shape}"
        )
        # TODO: rozpakuj wektor `w` na macierze W1, b1, W2, b2.
        # Przykład (NumPy, bez PyTorcha -- wystarczy do forward pass):
        i = 0
        self.W1 = w[i:i + _W1].reshape(HIDDEN_SIZE, STATE_VECTOR_SIZE); i += _W1
        self.b1 = w[i:i + _B1];                                         i += _B1
        self.W2 = w[i:i + _W2].reshape(NUM_ACTIONS, HIDDEN_SIZE);       i += _W2
        self.b2 = w[i:i + _B2];                                         i += _B2

    def __call__(self, card: int, history: str) -> Action:
        x = encode_state(card, history)            # wektor (12,)
        # TODO: forward pass. Przykład MLP z ReLU:
        h = np.maximum(0.0, self.W1 @ x + self.b1)  # warstwa ukryta
        logits = self.W2 @ h + self.b2              # warstwa wyjściowa (2,)
        return Action(int(np.argmax(logits)))
