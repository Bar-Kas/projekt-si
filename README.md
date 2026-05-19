# Kuhn Poker - Neural Network + Genetic Algorithm

Projekt z przedmiotu Sztuczna Inteligencja (gr. 4).

Implementacja agenta grającego w Kuhn Poker przy pomocy sieci neuronowej,
której wagi są optymalizowane algorytmem genetycznym (neuroewolucja).

## Wymagania

- Python 3.10 lub nowszy
- pip

## Instalacja

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## Uruchomienie testów

```bash
python -m pytest tests/ -v
```

## Uruchomienie treningu

```bash
python main.py
```

Po treningu w katalogu projektu pojawią się pliki:

- `best_weights.json` - wagi najlepszego osobnika
- `logbook.json` - statystyki każdej generacji (do wykresów)

## Struktura projektu

```
kuhn-poker-ga/
├── src/
│   ├── engine/
│   │   ├── kuhn.py          # zasady gry, terminalne historie, payoffs
│   │   └── state.py         # kodowanie stanu na wektor
│   ├── expert/
│   │   └── baseline.py      # system ekspertowy + Nash + random
│   └── ga/
│       ├── fitness.py       # funkcja fitness (mecze)
│       ├── linear_agent.py  # placeholder polityki liniowej
│       └── evolution.py     # pętla DEAP
├── tests/
│   └── test_engine.py       # testy jednostkowe
├── docs/
│   └── main.tex             # dokumentacja LaTeX (sekcje 1 i 2)
├── main.py                  # punkt wejścia
├── requirements.txt
└── README.md
```

## Podział pracy

- **Osoba 1**: silnik gry, GA, baseline ekspertowy, dokumentacja sekcje 1-2.
- **Osoba 2**: sieć neuronowa zastępująca `linear_agent_from_weights`,
  dokumentacja sekcje 3-4.

Interfejs między modułami: funkcja `(weights: list[float]) -> agent`,
gdzie `agent(card, history) -> Action`.
