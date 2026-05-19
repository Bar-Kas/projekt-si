import numpy as np

class StateEncoder:
    """
    Enkoder przekształca czytelny stan gry na wektor numeryczny 9D dla sieci neuronowej.
    """
    @staticmethod
    def encode(card: str, position: int, history: list[str]) -> np.ndarray:
        """
        :param card: "J", "Q" lub "K"
        :param position: 0 (Gracz 1) lub 1 (Gracz 2)
        :param history: lista akcji np. ["check", "bet"]
        :return: np.array o kształcie (9,)
        """
        # 1. Kodowanie karty (3 neurony)
        card_vec = [0, 0, 0]
        if card == "J":
            card_vec[0] = 1
        elif card == "Q":
            card_vec[1] = 1
        elif card == "K":
            card_vec[2] = 1

        #2. Kodowanie pozycji (1 neuron)
        pos_vec = [position]

        #3. Kodowanie historii licytacji (5 neuronów)
        #Indeksy:
        #0: Gracz 1 zrobił Check
        #1: Gracz 1 zrobił Bet
        #2: Gracz 2 zrobił Check (koniec)
        #3: Gracz 2 zrobił Bet/Call
        #4: Gracz 2 zrobił Fold (koniec)
        history_vec = [0, 0, 0, 0, 0]

        for i, action in enumerate(history):
            action_lower = action.lower()
            if i == 0 and position == 1:  #Pierwsza akcja gracza 1
                if action_lower in ["check", "ch"]:
                    history_vec[0] = 1
                elif action_lower in ["bet", "b"]:
                    history_vec[1] = 1

            elif i == 1:  #Odpowiedź gracza 2
                if action_lower in ["check", "ch", "call", "c"]:
                    history_vec[2] = 1  #Tu uproszczenie: call/check jako zgoda
                elif action_lower in ["bet", "b", "raise", "r"]:
                    history_vec[3] = 1
                elif action_lower in ["fold", "f"]:
                    history_vec[4] = 1

        #Połączenie wszystkiego w jeden wektor wejściowy (9 elementów)
        state_vector = card_vec + pos_vec + history_vec
        return np.array(state_vector, dtype=float)