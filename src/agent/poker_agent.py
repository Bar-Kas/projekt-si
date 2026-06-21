import numpy as np
from src.agent.network import PokerNetwork
from src.agent.encoder import StateEncoder
from src.engine.kuhn import Action, Card

class NNAgent:
    """
    Agent sieci neuronowej — adapter między silnikiem gry (int/str)
    a wewnętrzną reprezentacją sieci (str karta, int pozycja, list historia).
    """

    def __init__(self, chromosome: list[float] = None):
        self.network = PokerNetwork()
        self.encoder = StateEncoder()
        if chromosome is not None:
            self.network.set_weights(chromosome)

    def get_chromosome(self) -> list[float]:
        return self.network.get_weights().tolist()

    def set_chromosome(self, chromosome: list[float]):
        self.network.set_weights(chromosome)

    def __call__(self, card: int, history: str) -> Action:
        """Interfejs wymagany przez silnik gry: (card: int, history: str) -> Action."""
        card_str = {1: "J", 2: "Q", 3: "K"}.get(card, "J")
        position = len(history) % 2  # parzysta długość → ruch P1

        history_list = []
        for char in history:
            if char == "p":
                history_list.append("check")
            elif char == "b":
                history_list.append("bet")

        return Action(self.act(card_str, position, history_list))

    def get_probs(self, card: str, position: int, history: list[str]) -> np.ndarray:
        """Zwraca rozkład prawdopodobieństwa [p_check, p_bet] bez losowania."""
        state_vector = self.encoder.encode(card, position, history)
        outputs = self.network.forward(state_vector)
        total = np.sum(outputs)
        if total > 0:
            return outputs / total
        return np.array([0.5, 0.5])

    def act(self, card: str, position: int, history: list[str]) -> int:
        """Losuje akcję według rozkładu sieci (strategia mieszana)."""
        probs = self.get_probs(card, position, history)
        return int(np.random.choice([0, 1], p=probs))


def nn_agent_from_weights(weights):
    """Fabryka agenta — przekazywana do run_evolution jako agent_from_weights."""
    return NNAgent(weights)
