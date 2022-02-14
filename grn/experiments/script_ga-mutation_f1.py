#!/usr/bin/env python
# coding: utf-8

import os

from lib.score import score_both_size_new

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"

from model import Brain
from submodels import factories
import pandas as pd
from itertools import accumulate
import numpy as np
from lib.sde.grn.grn2 import GRNMain2
from lib.sde.mutate import multi_mutate_grn2
from lib.ga.utils import weighted_selection, normalize_fitness_values
from jf.utils.export import Exporter
from jf.autocompute.jf import O
from jf.models.stringmodel import read_model


REF = pd.read_csv("output/results/setup_basic/export/ref_basic2.csv")  # ref is a mean
SM_GEN = read_model("generation")

NB_GENES = 7

NAME = "mutation_f1"



# In[5]:


def individual_generator():
    return Solution(GRNMain2(NB_GENES, 0, 0))


class Solution:
    def __init__(self, grn):
        self.grn = grn
        
    def copy(self):
        return Solution(self.grn.copy())
        
    def mutate(self):
        multi_mutate_grn2(self.grn, value=1, method="fixed")


def run_grn(prun, grn):
    bb = get_bb(prun, grn)
    bb.run()
    return bb


def get_bb(prun, grn):
    ccls = factories["grn2_opti"](grn=grn)
    bb = Brain(time_step=0.5, verbose=False, start_population=5, max_pop_size=1e3,
            cell_cls=ccls, end_time=prun.end_time, start_time=50, silent=True)
    return bb


def fitness_func(prun, grn, score_func):
    bb = run_grn(prun, grn)
    output = score_func(bb.stats, REF, max_step=prun.end_time)
    fitness = 1.0 / output
    return fitness


def fitness_multistep(prun, grn, steps):
    total_fitness = 0
    stop = False
    previous_time = None
    bb = get_bb(prun, grn)
    # first step
    for step in steps:
        if not bb.run_until(step.end_time):
            stop = True
        score_step = score_both_size_new(bb.stats, prun.ref, max_step=step.end_time, min_step=previous_time)
        fitness_step = 1.0 / score_step
        fitness_step = min(fitness_step, step.max_fitness)
        total_fitness += fitness_step
        if fitness_step < step.min_fitness or stop:
            return total_fitness, bb.stats
        previous_time = step.end_time
        
    return total_fitness, bb.stats


def score_multistep(prun, stats, steps):
    total_fitness = 0
    stop = False
    previous_time = None
    # first step
    for step in steps:
        score_step = score_both_size(stats, prun.ref, max_step=step.end_time, min_step=previous_time)
        fitness_step = 1.0 / score_step
        fitness_step = min(fitness_step, step.max_fitness)
        total_fitness += fitness_step
        if fitness_step < step.min_fitness or stop:
            return total_fitness
        previous_time = step.end_time
        
    return total_fitness


def mean_sd_fitness(prun, grn, steps, run=3):
    fitnesses = [fitness_multistep(prun, grn, steps) for i in range(run)]
    return np.mean(fitnesses), np.std(fitnesses)


def do_init_pop(prun):
    return [individual_generator() for i in range(prun.pop_size)]


def do_fitness(prun, pop):
    # fitness = [fitness_func(prun, sol.grn, score_func) for sol in pop]
    # fitness = [fitness_strategy(prun, sol.grn) for sol in pop]
    fitness, stats = zip(*[fitness_multistep(prun, sol.grn, prun.steps) for sol in pop])
    return fitness, stats


def do_selection(prun, pop_fit, pop):
    # print("Fit score : ", pop_fit)
    acc = list(accumulate(pop_fit))
    best = max(pop_fit)
    best_id = pop_fit.index(best)
    
    print("Total fitness :", acc[-1])
    
    new_pop_fit = normalize_fitness_values(pop_fit)
    
    pop_sel, history_sel = weighted_selection(pop, new_pop_fit, individual_generator, new_fitness=0.3)
        
    return pop_sel, history_sel, best_id


def do_mutation(prun, pop_sel):
    [p.mutate() for p in pop_sel]
    return pop_sel


def pick_last_exported(exporter):
    generations = list(filter(SM_GEN.match, exporter.list()))
    if len(generations) == 0:
        return None, 0
    
    last = max(generations, key=lambda x: int(SM_GEN.extract(x).get("generation")))
    n_gen = int(SM_GEN.extract(last).get("generation")) + 1
    exporter.print(f"Found generation {n_gen - 1}", "reload")
    pop = exporter.load(last)["solution"]
    return pop, n_gen


def main(prun):
    exporter = Exporter(name=prun.name, copy_stdout=True)
    definition = """
    use sqrt for normalizing the values instead of abs
    """
    exporter.print(definition, slot="definition")
    best = 0
    pop, n_gen = pick_last_exported(exporter)
    if pop is None:
        pop = do_init_pop(prun)
        n_gen = 0
        
    for generation in range(n_gen, prun.n_gen):
        # args.generation = generation
        # objective.new_trial()
        fit, stats = do_fitness(prun, pop)
        # objective.best_current(max(fit))
        
        # TODO get the stats associated with the best scores
        sel, history_sel, best_id = do_selection(prun, fit, pop)
        if fit[best_id] > best:
            exporter.print(f"++ Best {fit[best_id]}")
            best = fit[best_id]
        else:
            exporter.print(f"-- Best {fit[best_id]}")
        pop = do_mutation(prun, sel)
        
        # history
        monitor = dict(
            transition=history_sel,
            solution=pop,
            fitness=fit,
            stats=stats,
        )
        exporter(monitor, SM_GEN.fill(generation=generation))
        
    return best


class ObjectiveStep(O):
    end_time = 0
    max_fitness = 4
    min_fitness = 0.75


example_steps = [
    ObjectiveStep(end_time=53),
    ObjectiveStep(end_time=56),
    ObjectiveStep(end_time=59),
    ObjectiveStep(end_time=62),
    ObjectiveStep(end_time=65),
    ObjectiveStep(end_time=68),
    ObjectiveStep(end_time=71),
    ObjectiveStep(end_time=74),
    ObjectiveStep(end_time=77),
    ObjectiveStep(end_time=80),
    ObjectiveStep(end_time=83),
    ObjectiveStep(end_time=86),
    ObjectiveStep(end_time=89),
]


class ParamRun(O):
    pop_size = 50
    n_gen = 100
    current_gen = 0
    end_time = 89
    ref = REF
    name = None


if __name__ == "__main__":
    args = ParamRun()
    args.steps = example_steps
    args.name = NAME
    main(args)