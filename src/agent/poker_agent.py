from src.agent.network import PokerNetwork
from src.agent.encoder import StateEncoder

class NNAgent:
    """
    Główny agent dla komunikacji.
    Ukrywa w sobie sieć neuronową i enkoder. Wystawia tylko prosty interfejs.
    """

    def __init__(self, chromosome: list[float] = None):
        """
        :param chromosome: Wektor 194 floatów z DEAP. Jeśli None, inicjuje pustą sieć.
        """
        self.network = PokerNetwork()
        self.encoder = StateEncoder()
        if chromosome is not None:
            self.network.set_weights(chromosome)

    def get_chromosome(self) -> list[float]:
        """Zwraca wagi agenta jako płaską listę (potrzebne Osobie 1 do DEAP)"""
        return self.network.get_weights().tolist()

    def set_chromosome(self, chromosome: list[float]):
        """Podmienia wagi agenta (potrzebne Osobie 1 w pętli ewolucji)"""
        self.network.set_weights(chromosome)

    def act(self, card: str, position: int, history: list[str]) -> int:
        """
        Główna metoda dla Osoby 1. Jej silnik wywołuje to podczas gry.
        :return: 0 (Check/Fold) lub 1 (Bet/Call)
        """
        # 1. Enkoder zamienia czytelne dane na wektor 9D
        state_vector = self.encoder.encode(card, position, history)

        # 2. Sieć podejmuje decyzję na podstawie wektora
        action = self.network.get_action(state_vector)

        return action