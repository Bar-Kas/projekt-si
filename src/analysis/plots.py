import json
import os
import numpy as np
import matplotlib.pyplot as plt
from src.training.metrics import NASH_REFERENCE


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

# Kolejność sytuacji decyzyjnych i ich etykiety na wykresach
_SITUATION_KEYS = [
    'P1_Jack_Open', 'P1_Queen_Open', 'P1_King_Open', 'P1_Queen_Facing_Bet',
    'P2_Jack_After_Check', 'P2_Queen_After_Check', 'P2_King_After_Check',
    'P2_Jack_After_Bet', 'P2_Queen_After_Bet', 'P2_King_After_Bet',
]
_SITUATION_LABELS = [
    'P1/J\notwiera', 'P1/Q\notwiera', 'P1/K\notwiera', 'P1/Q\nvs bet',
    'P2/J\npo czek', 'P2/Q\npo czek', 'P2/K\npo czek',
    'P2/J\nvs bet',  'P2/Q\nvs bet',  'P2/K\nvs bet',
]

# Mapowanie kluczy GA profile → kluczy CFR strategy
_CFR_KEY_MAP = {
    'P1_Jack_Open':        (1, ''),
    'P1_Queen_Open':       (2, ''),
    'P1_King_Open':        (3, ''),
    'P1_Queen_Facing_Bet': (2, 'pb'),
    'P2_Jack_After_Check': (1, 'p'),
    'P2_Queen_After_Check':(2, 'p'),
    'P2_King_After_Check': (3, 'p'),
    'P2_Jack_After_Bet':   (1, 'b'),
    'P2_Queen_After_Bet':  (2, 'b'),
    'P2_King_After_Bet':   (3, 'b'),
}


def plot_learning_curve(
    logbook_path: str = "logbook.json",
    save_path: str = "grafs/learning_curve.png",
) -> None:
    """Wykres zbieżności GA: avg/max/min fitness przez generacje."""
    try:
        with open(logbook_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Błąd: nie znaleziono {logbook_path}. Uruchom najpierw main.py.")
        return

    generations  = [row["gen"] for row in data]
    avg_fitness  = [row["avg"] for row in data]
    max_fitness  = [row["max"] for row in data]
    min_fitness  = [row["min"] for row in data]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(generations, max_fitness, label="Maksymalny fitness", color="green", linewidth=2)
    ax.plot(generations, avg_fitness, label="Średni fitness populacji", color="royalblue",
            linestyle="--", linewidth=1.5)
    ax.fill_between(generations, min_fitness, max_fitness,
                    color="royalblue", alpha=0.08, label="Zakres (min–max)")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_title("Zbieżność algorytmu genetycznego — Kuhn Poker", fontsize=13, fontweight="bold")
    ax.set_xlabel("Generacja", fontsize=11)
    ax.set_ylabel("Fitness (średni zysk chipów / rozdanie)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300)
    plt.show()
    print(f"Zapisano: {save_path}")


def plot_strategy_profile(
    ga_profile: dict,
    cfr_strategy: dict,
    save_path: str = "grafs/strategy_profile.png",
) -> None:
    """Grouped bar chart: GA champion vs CFR vs Nash dla każdej sytuacji decyzyjnej."""
    ga_vals   = [ga_profile[k] for k in _SITUATION_KEYS]
    cfr_vals  = [cfr_strategy.get(_CFR_KEY_MAP[k], [0.5, 0.5])[1] for k in _SITUATION_KEYS]  # [1] = p_bet
    nash_vals = [NASH_REFERENCE[k] for k in _SITUATION_KEYS]

    x = np.arange(len(_SITUATION_KEYS))
    w = 0.26

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x - w,   ga_vals,   w, label="GA Champion",        color="steelblue",  alpha=0.85)
    ax.bar(x,       cfr_vals,  w, label="CFR (Nash approx.)", color="darkorange", alpha=0.85)
    ax.bar(x + w,   nash_vals, w, label="Nash Equilibrium",   color="seagreen",   alpha=0.85)

    # Pionowa linia separująca P1 i P2
    ax.axvline(x=3.5, color="grey", linewidth=1, linestyle="--", alpha=0.6)
    ax.text(1.5,  1.05, "Gracz 1 (inicjuje)", ha="center", fontsize=10, color="grey")
    ax.text(6.5,  1.05, "Gracz 2 (odpowiada)", ha="center", fontsize=10, color="grey")

    ax.set_xticks(x)
    ax.set_xticklabels(_SITUATION_LABELS, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("P(BET / CALL)", fontsize=11)
    ax.set_title("Profil strategiczny: GA Champion vs CFR vs Nash Equilibrium",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300)
    plt.show()
    print(f"Zapisano: {save_path}")
