from src.agent.poker_agent import NNAgent

# Nash equilibrium (alpha=1/6) jako punkt odniesienia
NASH_REFERENCE = {
    'P1_Jack_Open':        1/6,
    'P1_Queen_Open':       0.0,
    'P1_King_Open':        1/2,
    'P1_Queen_Facing_Bet': 1/2,
    'P2_Jack_After_Check': 1/3,
    'P2_Queen_After_Check': 0.0,
    'P2_King_After_Check': 1.0,
    'P2_Jack_After_Bet':   0.0,
    'P2_Queen_After_Bet':  1/3,
    'P2_King_After_Bet':   1.0,
}

def profile_agent_strategy(agent: NNAgent) -> dict:
    """
    Zwraca prawdopodobieństwo zagrania BET/CALL w kluczowych sytuacjach.
    Wartości są deterministyczne (nie losowe próbki).
    """
    def p_bet(card, position, history):
        return agent.get_probs(card=card, position=position, history=history)[1]

    return {
        'P1_Jack_Open':         p_bet("J", 0, []),
        'P1_Queen_Open':        p_bet("Q", 0, []),
        'P1_King_Open':         p_bet("K", 0, []),
        'P1_Queen_Facing_Bet':  p_bet("Q", 0, ["check", "bet"]),
        'P2_Jack_After_Check':  p_bet("J", 1, ["check"]),
        'P2_Queen_After_Check': p_bet("Q", 1, ["check"]),
        'P2_King_After_Check':  p_bet("K", 1, ["check"]),
        'P2_Jack_After_Bet':    p_bet("J", 1, ["bet"]),
        'P2_Queen_After_Bet':   p_bet("Q", 1, ["bet"]),
        'P2_King_After_Bet':    p_bet("K", 1, ["bet"]),
    }

def print_strategy_report(profile: dict):
    """Wyświetla profil jako prawdopodobieństwa BET, z porównaniem do Nash (alpha=1/6)."""
    def bar(p):
        filled = round(p * 20)
        return f"[{'#' * filled}{'.' * (20 - filled)}] {p*100:5.1f}%"

    def row(label, key):
        p = profile[key]
        nash = NASH_REFERENCE.get(key, 0.0)
        diff = abs(p - nash)
        nash_str = f"Nash: {nash*100:.0f}%"
        diff_str = f"delta={diff*100:.1f}%"
        print(f"  {label:<32} {bar(p)}  ({nash_str}  {diff_str})")

    print("\n" + "=" * 80)
    print("   PROFIL STRATEGICZNY — prawdopodobieństwo BET/CALL   ")
    print("=" * 80)
    print("Gracz 1 (zaczyna):")
    row("J  — blef?",              'P1_Jack_Open')
    row("Q  — bet?",               'P1_Queen_Open')
    row("K  — value bet",          'P1_King_Open')
    row("Q  vs bet (call/fold?)",  'P1_Queen_Facing_Bet')
    print("Gracz 2 (odpowiada):")
    row("J  po cheku",             'P2_Jack_After_Check')
    row("Q  po cheku",             'P2_Queen_After_Check')
    row("K  po cheku",             'P2_King_After_Check')
    row("J  vs bet",               'P2_Jack_After_Bet')
    row("Q  vs bet",               'P2_Queen_After_Bet')
    row("K  vs bet",               'P2_King_After_Bet')
    print("=" * 80)