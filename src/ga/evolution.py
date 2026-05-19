"""Genetic algorithm using DEAP to evolve a Kuhn Poker agent.

The individual genome is a flat list of floats. An `agent_from_weights`
function (provided by the caller) turns the genome into a playable agent.
This decoupling means the same evolution loop can drive a linear policy
today and a neural network tomorrow.
"""

from __future__ import annotations

import random

import numpy as np
from deap import algorithms, base, creator, tools

from src.expert.baseline import always_bet_agent, expert_agent, random_agent
from src.ga.fitness import evaluate_against_pool

# DEAP requires the classes to be defined at module level once.
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def make_toolbox(weight_size: int, agent_from_weights, eval_hands: int = 200):
    """Build a DEAP toolbox for the given genome size and agent factory."""
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

    baseline_pool = [expert_agent, random_agent, always_bet_agent]

    def eval_individual(individual):
        agent = agent_from_weights(individual)
        score = evaluate_against_pool(agent, baseline_pool, num_hands_each=eval_hands)
        return (score,)

    toolbox.register("evaluate", eval_individual)
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
    """Run the GA. Returns (best_individual, logbook)."""
    random.seed(seed)
    np.random.seed(seed)

    toolbox = make_toolbox(weight_size, agent_from_weights, eval_hands=eval_hands)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("std", np.std)

    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
    )

    return hof[0], log
