import numpy as np

class PokerNetwork:
    def __init__(self, input_size=10, hidden_size=16, output_size=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w1_shape = (input_size, hidden_size)
        self.b1_shape = (hidden_size,)
        self.w2_shape = (hidden_size, output_size)
        self.b2_shape = (output_size,)

        self.total_weights = (
            (input_size * hidden_size) + hidden_size +
            (hidden_size * output_size) + output_size
        )  # rozmiar chromosomu przekazywanego przez GA

        # Zera jako placeholder — GA nadpisze wagi chromosomem przed każdą oceną
        self.W1 = np.zeros(self.w1_shape)
        self.b1 = np.zeros(self.b1_shape)
        self.W2 = np.zeros(self.w2_shape)
        self.b2 = np.zeros(self.b2_shape)

    def set_weights(self, flat_vector):
        """Rekonstruuje macierze sieci z płaskiego wektora chromosomu DEAP."""
        if len(flat_vector) != self.total_weights:
            raise ValueError(
                f"Nieprawidłowa długość chromosomu! "
                f"Oczekiwano {self.total_weights}, podano {len(flat_vector)}"
            )
        idx = 0
        w1_len = self.input_size * self.hidden_size
        self.W1 = np.array(flat_vector[idx:idx + w1_len]).reshape(self.w1_shape)
        idx += w1_len

        b1_len = self.hidden_size
        self.b1 = np.array(flat_vector[idx:idx + b1_len])
        idx += b1_len

        w2_len = self.hidden_size * self.output_size
        self.W2 = np.array(flat_vector[idx:idx + w2_len]).reshape(self.w2_shape)
        idx += w2_len

        self.b2 = np.array(flat_vector[idx:idx + self.b2_shape[0]])

    def get_weights(self):
        """Spłaszcza wagi i biasy do jednego wektora 1D."""
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten()
        ])

    def forward(self, x):
        h = np.maximum(0, np.dot(x, self.W1) + self.b1)           # ReLU
        z = np.clip(np.dot(h, self.W2) + self.b2, -500, 500)      # clip zapobiega overflow w exp
        return 1 / (1 + np.exp(-z))                                # Sigmoid → [0, 1]

    def get_action(self, state_vector):
        """Zwraca akcję z najwyższym prawdopodobieństwem (0: check/fold, 1: bet/call)."""
        return np.argmax(self.forward(state_vector))
