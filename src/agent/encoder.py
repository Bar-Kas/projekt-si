import numpy as np

class StateEncoder:
    """
    Enkoder przekształca stan gry na wektor numeryczny 10D dla sieci neuronowej.

    Struktura wektora (10 elementów):
      [0-2]  karta (one-hot: J, Q, K)
      [3]    pozycja (0 = Gracz 1, 1 = Gracz 2)
      [4-8]  historia akcji (5 neuronów, patrz niżej)
      [9]    zarezerwowane / padding (zawsze 0)
    """

    @staticmethod
    def encode(card: str, position: int, history: list[str]) -> np.ndarray:
        """
        :param card: "J", "Q" lub "K"
        :param position: 0 (Gracz 1) lub 1 (Gracz 2)
        :param history: lista akcji np. ["check", "bet"]
        :return: np.array o kształcie (10,)
        """
        # Kodowanie karty — one-hot (J=0, Q=1, K=2)
        card_vec = [0, 0, 0]
        if card == "J":
            card_vec[0] = 1
        elif card == "Q":
            card_vec[1] = 1
        elif card == "K":
            card_vec[2] = 1

        # Pozycja gracza (0 lub 1)
        pos_vec = [position]

        # Historia licytacji — 5 neuronów:
        #   [0] P1 sprawdził (check)
        #   [1] P1 postawił (bet)
        #   [2] P2 sprawdził / spasował (check/fold po becie P1)
        #   [3] P2 postawił / sprawdził (bet/call)
        #   [4] P2 spasował (fold)
        history_vec = [0, 0, 0, 0, 0]

        for i, action in enumerate(history):
            action_lower = action.lower()
            if i == 0:  # pierwsza akcja: ruch P1
                if action_lower in ["check", "ch"]:
                    history_vec[0] = 1
                elif action_lower in ["bet", "b"]:
                    history_vec[1] = 1

            elif i == 1:  # druga akcja: odpowiedź P2
                if action_lower in ["check", "ch", "call", "c"]:
                    history_vec[2] = 1
                elif action_lower in ["bet", "b", "raise", "r"]:
                    history_vec[3] = 1
                elif action_lower in ["fold", "f"]:
                    history_vec[4] = 1

        # Padding: dodatkowy neuron wyrównuje wektor do 10 elementów (input_size sieci)
        state_vector = card_vec + pos_vec + history_vec + [0]
        return np.array(state_vector, dtype=float)
