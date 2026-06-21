"""
Counterfactual Regret Minimization (CFR) dla Kuhn Poker.

Algorytm zaprojektowany do gier z niepełną informacją — matematycznie
gwarantuje zbieżność do równowagi Nasha. Służy jako punkt odniesienia
dla agenta GA+NN.

Zbiega do Nasha w ~1000 iteracjach dla Kuhn Poker (12 information sets).
"""
from __future__ import annotations

import numpy as np
from itertools import permutations

from src.engine.kuhn import Action

# J=1, Q=2, K=3
CARDS = [1, 2, 3]
CARD_NAMES = {1: 'J', 2: 'Q', 3: 'K'}

# Wszystkie możliwe zakończenia gry
_TERMINAL = {'pp', 'bp', 'bb', 'pbp', 'pbb'}


def _is_terminal(history: str) -> bool:
    return history in _TERMINAL


def _payoff(cards: tuple[int, int], history: str) -> float:
    """Wypłata dla gracza 0 (P1) w chipach netto."""
    c0, c1 = cards
    showdown = 1.0 if c0 > c1 else -1.0

    if history == 'pp':  return showdown        # oba check, showdown
    if history == 'bp':  return 1.0             # P2 sfoldował
    if history == 'bb':  return 2.0 * showdown  # oba bet, showdown
    if history == 'pbp': return -1.0            # P1 sfoldował
    if history == 'pbb': return 2.0 * showdown  # P1 call po P2 bet, showdown
    raise ValueError(f"Nieznana historia terminalna: {history}")


def _regret_match(regrets: list[float]) -> list[float]:
    """Regret matching — strategia proporcjonalna do dodatnich regretów."""
    positive = [max(r, 0.0) for r in regrets]
    total = sum(positive)
    if total > 0:
        return [p / total for p in positive]
    return [0.5, 0.5]  # uniform gdy brak regretów


def _cfr(
    cards: tuple[int, int],
    history: str,
    p0: float,
    p1: float,
    regret_sum: dict,
    strategy_sum: dict,
) -> float:
    """
    Rekurencyjny CFR. Zwraca oczekiwaną wypłatę dla gracza 0.

    p0, p1 — reach probability każdego gracza od korzenia do bieżącego węzła.
    """
    if _is_terminal(history):
        return _payoff(cards, history)

    player = len(history) % 2   # 0 = P1 (długości parzyste), 1 = P2
    card = cards[player]
    info_set = (card, history)

    strategy = _regret_match(regret_sum.get(info_set, [0.0, 0.0]))

    # Sumuj strategię ważoną własnym reach → służy do wyznaczenia avg strategy
    own_reach = p0 if player == 0 else p1
    ss = strategy_sum.setdefault(info_set, [0.0, 0.0])
    ss[0] += own_reach * strategy[0]
    ss[1] += own_reach * strategy[1]

    # Oblicz wypłatę dla każdej akcji
    util = [0.0, 0.0]
    for a, action in enumerate(['p', 'b']):
        new_history = history + action
        if player == 0:
            util[a] = _cfr(cards, new_history, p0 * strategy[a], p1,
                           regret_sum, strategy_sum)
        else:
            util[a] = _cfr(cards, new_history, p0, p1 * strategy[a],
                           regret_sum, strategy_sum)

    # Oczekiwana wypłata przy bieżącej strategii
    node_util = strategy[0] * util[0] + strategy[1] * util[1]

    # Counterfactual regret update
    opp_reach = p1 if player == 0 else p0
    rs = regret_sum.setdefault(info_set, [0.0, 0.0])
    for a in range(2):
        if player == 0:
            # P0 maksymalizuje → regret = (util akcji a) - (expected util)
            rs[a] += opp_reach * (util[a] - node_util)
        else:
            # P1 minimalizuje P0's utility → P1's regret = node_util - util[a]
            rs[a] += opp_reach * (node_util - util[a])

    return node_util


def train(iterations: int = 10_000) -> dict:
    """
    Trenuje CFR przez podaną liczbę iteracji.

    Zwraca uśrednioną strategię Nash:
        {(card: int, history: str): [p_check, p_bet]}
    """
    regret_sum: dict = {}
    strategy_sum: dict = {}

    for _ in range(iterations):
        for cards in permutations(CARDS, 2):
            _cfr(cards, '', 1.0, 1.0, regret_sum, strategy_sum)

    avg_strategy: dict = {}
    for info_set, ss in strategy_sum.items():
        total = ss[0] + ss[1]
        avg_strategy[info_set] = (
            [ss[0] / total, ss[1] / total] if total > 0 else [0.5, 0.5]
        )
    return avg_strategy


class CfrAgent:
    """Agent grający zgodnie z wyuczoną strategią CFR."""

    def __init__(self, strategy: dict):
        self.strategy = strategy

    def __call__(self, card: int, history: str) -> Action:
        probs = self.strategy.get((card, history), [0.5, 0.5])
        return Action(int(np.random.choice([0, 1], p=probs)))


# Teoretyczna równowaga Nasha (alpha=1/6) jako referencja
NASH_REFERENCE = {
    (1, ''):   1/6,   (2, ''):   0.0,  (3, ''):   1/2,
    (2, 'pb'): 1/2,
    (1, 'p'):  1/3,   (2, 'p'):  0.0,  (3, 'p'):  1.0,
    (1, 'b'):  0.0,   (2, 'b'):  1/3,  (3, 'b'):  1.0,
}

_PROFILE_ROWS = [
    ("P1 J  — blef?",      (1, ''),   "← Nash: 17%"),
    ("P1 Q  — bet?",       (2, ''),   "← Nash:  0%"),
    ("P1 K  — value bet",  (3, ''),   "← Nash: 50%"),
    ("P1 Q  vs P2 bet",    (2, 'pb'), "← Nash: 50%"),
    ("P2 J  po cheku",     (1, 'p'),  "← Nash: 33%"),
    ("P2 Q  po cheku",     (2, 'p'),  "← Nash:  0%"),
    ("P2 K  po cheku",     (3, 'p'),  "← Nash:100%"),
    ("P2 J  vs bet",       (1, 'b'),  "← Nash:  0%"),
    ("P2 Q  vs bet",       (2, 'b'),  "← Nash: 33%"),
    ("P2 K  vs bet",       (3, 'b'),  "← Nash:100%"),
]


def print_cfr_strategy(strategy: dict) -> None:
    """Wyświetla profil strategiczny CFR z porównaniem do Nasha."""

    def bar(p: float) -> str:
        n = round(p * 20)
        return f"[{'#' * n}{'.' * (20 - n)}] {p * 100:5.1f}%"

    print("\n" + "=" * 76)
    print("   CFR — wyuczona strategia (zbieżność do Nasha)")
    print("=" * 76)
    total_error = 0.0
    for label, key, note in _PROFILE_ROWS:
        p_bet = strategy.get(key, [0.5, 0.5])[1]
        nash = NASH_REFERENCE[key]
        diff = abs(p_bet - nash)
        total_error += diff
        print(f"  {label:<22} {bar(p_bet)}  {note}  delta={diff * 100:.1f}%")
    print(f"\n  Średni błąd od Nasha: {total_error / len(_PROFILE_ROWS) * 100:.2f}%")
    print("=" * 76)
