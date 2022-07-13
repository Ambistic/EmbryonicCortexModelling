#!/usr/bin/env python
# coding: utf-8

import os

from lib.score import score_both_size_norm

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from model import Brain
from submodels import factories
import pandas as pd
from itertools import accumulate
import numpy as np
import argparse
from lib.sde.grn.grn3 import GRNMain3 as GRNMain
from lib.sde.mutate import multi_mutate_grn2
from lib.ga.utils import weighted_selection
from jf.utils.export import Exporter
from jf.autocompute.jf import O
from jf.models.stringmodel import read_model


REF = pd.read_csv("output/results/setup_basic/export/ref_basic2.csv")  # ref is a mean
SM_GEN = read_model("generation")
VALUE_MUTATION = 2


def individual_generator():
    return Solution(GRNMain(NB_GENES, 0, 0))


class Solution:
    def __init__(self, grn):
        self.grn = grn
        
    def copy(self):
        return Solution(self.grn.copy())
        
    def mutate(self):
        # TODO add poisson number of mutation
        multi_mutate_grn2(self.grn, value=VALUE_MUTATION, method="poisson")


def run_grn(prun, grn):
    bb = get_bb(prun, grn)
    bb.run()
    return bb


def get_bb(prun, grn):
    ccls = factories["grn3"](grn=grn)
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
        score_step = score_both_size_norm(bb.stats, prun.ref, max_step=step.end_time, min_step=previous_time, norm=prun.norm)
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
        score_step = score_both_size_norm(stats, prun.ref, max_step=step.end_time, min_step=previous_time, norm=prun.norm)
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
    fitness, stats = zip(*[fitness_multistep(prun, sol.grn, prun.steps) for sol in pop])
    return fitness, stats


def do_selection(prun, pop_fit, pop):
    # print("Fit score : ", pop_fit)
    acc = list(accumulate(pop_fit))
    best = max(pop_fit)
    best_id = pop_fit.index(best)
    
    print("Total fitness :", acc[-1])
    
    pop_sel, history_sel = weighted_selection(pop, pop_fit, individual_generator, new_fitness=0.3)
        
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
    first change on using absolute normalization
    early working set up was using same as power 2 normalisation
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


class ParamRun(O):
    pop_size = 50
    n_gen = 100
    current_gen = 0
    end_time = 89
    ref = REF
    name = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", type=int, default=7,
                        help="Number of genes")
    parser.add_argument("-t", "--threshold", type=float, default=1.0,
                        help="Min threshold for objective step")
    parser.add_argument("-m", "--mutations", type=int, default=2,
                        help="Value for mutation function")
    parser.add_argument("--norm", type=float, default=1.,
                        help="Normalisation value for size scoring")
    parser.add_argument("-d", "--date", type=str, default="",
                        help="Date for naming")
    parser.add_argument("--name", type=str, required=False, default=None,
                        help="Name to be saved")
    parser.add_argument("-g", "--generations", type=int, default=50,
                        help="Number of generations to do")
    parser.add_argument("-i", "--id", type=int, default=1,
                        help="Id if multiple trials")
    pargs = parser.parse_args()
    
    if pargs.name is None:
        pargs.name = f"result_m{pargs.mutations}_t{pargs.threshold}_n{pargs.number}_o{pargs.norm}_i{pargs.id}"
    
    NB_GENES = pargs.number
    VALUE_MUTATION = pargs.mutations
    
    args = ParamRun()
    args.n_gen = pargs.generations
    args.norm = pargs.norm
    
    class ObjectiveStep(O):
        end_time = 0
        max_fitness = 4
        min_fitness = pargs.threshold


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
    
    args.steps = example_steps
    args.name = pargs.name
    main(args)