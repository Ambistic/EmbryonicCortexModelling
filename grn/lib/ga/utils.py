import numpy as np
from itertools import accumulate


def get_sample_index(sample, acc):
    val = min(filter(lambda x: x > sample, acc))
    return acc.index(val)


def weighted_selection(population, fitness_values, create_func, new_fitness=1.):
    """
    :param population: list of individuals
    :param fitness_values: fitness values for each individual (same length with population)
    :param create_func: function to create new individual
    :param new_fitness: fitness value for a new solution

    :returns: a new population with the same size as the population argument
    """
    pop_sel = []
    history_sel = []
    acc = list(accumulate(fitness_values))
    total_population_fitness = acc[-1]
    total_fitness = total_population_fitness + new_fitness
    for i in range(len(population)):
        sample = np.random.uniform(0, total_fitness)
        if sample >= total_population_fitness:
            pop_sel.append(create_func())
            history_sel.append(-1)
        else:
            chosen_id = get_sample_index(sample, acc)
            pop_sel.append(population[chosen_id].copy())
            history_sel.append(chosen_id)

    return pop_sel, history_sel


def normalize_fitness_values(fitness_values: list, percentile=50):
    fitness_values = np.asarray(fitness_values)
    threshold = np.percentile(fitness_values, percentile)
    new_fitness_values = np.maximum(fitness_values - threshold, 0)
    return list(new_fitness_values)


def weighted_selection_one(population, fitness_values, create_func, new_fitness=1., id_=0):
    total_population_fitness = sum(fitness_values)
    acc = list(accumulate(fitness_values))
    total_fitness = total_population_fitness + new_fitness

    sample = np.random.uniform(0, total_fitness)
    if sample >= total_population_fitness:
        return create_func(id_), -1
    else:
        chosen_id = get_sample_index(sample, acc)
        return population[chosen_id].copy(id_), chosen_id
