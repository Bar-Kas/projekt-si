import random
import numpy as np
from src.agent.network import PokerNetwork

def test_network_serialization():
    net = PokerNetwork()
    #Sprawdzenie matematycznej wielkości sieci
    assert net.total_weights == 194

    #Używamy czystego Pythona do generowania listy - PyCharm jest szczęśliwy
    mock_chromosom = [random.uniform(-1.0, 1.0) for _ in range(194)]

    #Wstrzykujemy wagi do sieci
    net.set_weights(mock_chromosom)

    #Wyciągamy je z powrotem i sprawdzamy czy są identyczne
    flat_back = net.get_weights()
    assert flat_back.shape == (194,)
    assert np.allclose(mock_chromosom, flat_back)

def test_forward_pass_execution():
    net = PokerNetwork()
    mock_chromosom = [random.uniform(-0.5, 0.5) for _ in range(194)]
    net.set_weights(mock_chromosom)

    #Tworzymy sztuczny, zakodowany wektor stanu wejściowego (9 elementów)
    mock_input_state = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])

    #Sprawdzamy wyjście sieci
    probabilities = net.forward(mock_input_state)

    assert probabilities.shape == (2,)
    #Sigmoid musi zwracać wartości z przedziału [0, 1]
    assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0)

    #Sprawdzamy czy metoda decyzji zwraca poprawną akcję (0 lub 1)
    action = net.get_action(mock_input_state)
    assert action in [0, 1]