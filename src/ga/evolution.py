"""Genetic algorithm using DEAP to evolve a Kuhn Poker agent via Self-Play."""

from __future__ import annotations

import random

import numpy as np
from deap import base, creator, tools

from src.ga.fitness import play_match
from src.expert.baseline import nash_agent_factory, expert_agent

# Waga regularyzacji entropii — nagradza strategie mieszane zamiast 0%/100%.
# Nash ≈ 0.31 bit avg entropy → +0.012 bonus vs deterministyczny 0.
ENTROPY_WEIGHT = 0.06

# Te same 10 sytuacji decyzyjnych co w profilu strategicznym (metrics.py)
_ENTROPY_SITUATIONS = [
    ("J", 0, []),              ("Q", 0, []),              ("K", 0, []),
    ("Q", 0, ["check", "bet"]),
    ("J", 1, ["check"]),       ("Q", 1, ["check"]),       ("K", 1, ["check"]),
    ("J", 1, ["bet"]),         ("Q", 1, ["bet"]),         ("K", 1, ["bet"]),
]


def _agent_entropy(agent) -> float:
    """Średnia entropia Shannona [bit] agenta w 10 sytuacjach decyzyjnych.
    Max = 1 bit (50-50), Nash ≈ 0.31 bit, deterministyczny ≈ 0 bit."""
    total = 0.0
    for card, pos, hist in _ENTROPY_SITUATIONS:
        p = agent.get_probs(card=card, position=pos, history=hist)
        total += -sum(pi * np.log2(pi + 1e-10) for pi in p)
    return total / len(_ENTROPY_SITUATIONS)


# DEAP wymaga, aby te klasy były zdefiniowane na poziomie modułu
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def make_toolbox(weight_size: int, agent_from_weights, eval_hands: int = 200):
    """Buduje toolbox DEAP. 

    W wersji Self-Play nie rejestrujemy tutaj funkcji 'evaluate', 
    ponieważ ocena osobników zależy od dynamicznie zmieniającej się populacji.
    """
    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.uniform, -1.0, 1.0)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=weight_size,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def run_evolution(
    weight_size: int,
    agent_from_weights,
    pop_size: int = 60,
    generations: int = 100,
    cxpb: float = 0.7,
    mutpb: float = 0.2,
    eval_hands: int = 200,
    seed: int = 42,
    verbose: bool = True,
):
    """Uruchamia algorytm genetyczny w trybie Self-Play z zewnętrznymi opponentami."""
    random.seed(seed)
    np.random.seed(seed)

    toolbox = make_toolbox(weight_size, agent_from_weights, eval_hands=eval_hands)
    pop = toolbox.population(n=pop_size)
    # HOF(20) zbiera 20 kandydatów przez cały trening; re-ewaluacja na końcu
    # eliminuje szum (agenci mieli szczęście podczas self-play)
    hof = tools.HallOfFame(20)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("std", np.std)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields

    fixed_opponents = [
        nash_agent_factory(alpha=1 / 6, seed=seed),
        expert_agent,
    ]
    # 1 mecz self-play (diversity) + 2 fixed (stabilny sygnał, 2/3 wagi)
    num_self_play_opponents = 1

    if verbose:
        print("=== URUCHAMIANIE TRENINGU SELF-PLAY ===")

    for gen in range(0, generations + 1):
        current_agents = [agent_from_weights(ind) for ind in pop]

        for i, ind in enumerate(pop):
            other_indices = [j for j in range(len(pop)) if j != i]
            sampled_indices = random.sample(other_indices, num_self_play_opponents)

            scores = []
            for opp_idx in sampled_indices:
                score = play_match(current_agents[i], current_agents[opp_idx], num_hands=eval_hands)
                scores.append(score)

            for fixed_opp in fixed_opponents:
                score = play_match(current_agents[i], fixed_opp, num_hands=eval_hands)
                scores.append(score)

            entropy_bonus = _agent_entropy(current_agents[i])
            ind.fitness.values = (np.mean(scores) + ENTROPY_WEIGHT * entropy_bonus,)

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(pop), **record)
        if verbose:
            print(logbook.stream)

        if gen == generations:
            break

        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values   # wymóg DEAP: usuń stary wynik, by wymusić re-ewaluację
                del ind2.fitness.values

        for ind in offspring:
            if random.random() < mutpb:
                toolbox.mutate(ind)
                del ind.fitness.values

        pop[:] = offspring

    champion, best_score = hof[0], float('-inf')
    for cand in hof:
        agent = agent_from_weights(cand)
        game_score = np.mean([play_match(agent, opp, num_hands=2000) for opp in fixed_opponents])
        score = game_score + ENTROPY_WEIGHT * _agent_entropy(agent)
        if score > best_score:
            best_score, champion = score, cand
    return champion, logbook