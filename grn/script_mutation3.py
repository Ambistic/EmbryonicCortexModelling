#!/usr/bin/env python
# coding: utf-8

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from brain import BrainModel
from submodels import factories
import pandas as pd
import argparse
from lib.sde.grn.grn3 import GRNMain3 as GRNMain
from lib.sde.mutate import mutate_grn3
from lib.ga.utils import weighted_selection_one
from jf.utils.export import Exporter
from jf.utils.helper import provide_id
from jf.autocompute.jf import O
from jf.models.stringmodel import read_model
from lib.callback import (
    cell_number_callback, progenitor_number_callback, neuron_number_callback
)
from lib.score import (
    score_stats_norm
)


REF = O(
    stats=pd.read_csv("output/results/setup_basic/export/ref_basic2.csv")  # ref is a mean
)
SM_GEN = read_model("generation")
VALUE_MUTATION = 2


def individual_generator(id_=-1):
    return Solution(GRNMain(NB_GENES, 0, 0), id_=id_)


class Solution:
    def __init__(self, grn, id_=0, parent=-1):
        self.id = id_
        self.grn = grn
        self.parent = parent
        self.fit = -1
        self.stats = None
        
    def copy(self, id_=0):
        return Solution(self.grn.copy(), id_=id_, parent=self.id)
        
    def mutate(self):
        mutate_grn3(self.grn)

        
def score_bb_size(bb, ref, *args, **kwargs):
    return (
        score_stats_norm(bb.stats, ref.stats, col_stats=f"progenitor_pop_size",
                         col_ref="progenitor_pop_size", norm=2.0, *args, **kwargs)
        + score_stats_norm(bb.stats, ref.stats, col_stats=f"whole_pop_size",
                         col_ref="whole_pop_size", norm=2.0, *args, **kwargs)
    )


def run_grn(prun, grn):
    bb = get_bb(prun, grn)
    bb.run()
    return bb


def get_bb(prun, grn):
    ccls = factories["grn3"](grn=grn)
    callbacks = dict(
        progenitor_pop_size=progenitor_number_callback,
        whole_pop_size=cell_number_callback,
        neuron_pop_size=neuron_number_callback,
    )
    bb = BrainModel(time_step=0.5, verbose=False, start_population=7, max_pop_size=2e3,
            cell_cls=ccls, end_time=prun.end_time, start_time=50, silent=True,
              run_tissue=False, monitor_callbacks=callbacks)
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
        # score_step = score_both_size(bb.stats, prun.ref, max_step=step.end_time, min_step=previous_time)
        score_step = step.score_func(bb, prun.ref, max_step=step.end_time, min_step=step.start_time)
        fitness_step = 1.0 / score_step
        fitness_step = min(fitness_step, step.max_fitness)
        total_fitness += fitness_step
        if fitness_step < step.min_fitness or stop:
            return total_fitness, bb.stats
        else:
            previous_time = step.end_time
            step.passed()
        
    return total_fitness, bb.stats


def score_multistep(prun, bb, steps):
    total_fitness = 0
    stop = False
    previous_time = None
    # first step
    for step in steps:
        score_step = step.score_func(bb, prun.ref, max_step=step.end_time, min_step=previous_time, norm=prun.norm)
        fitness_step = 1.0 / score_step
        fitness_step = min(fitness_step, step.max_fitness)
        total_fitness += fitness_step
        if fitness_step < step.min_fitness or stop:
            return total_fitness
        previous_time = step.end_time
        
    return total_fitness


def do_init(prun):
    return individual_generator(provide_id())

def do_fitness(prun, sol):
    fitness, stats = fitness_multistep(prun, sol.grn, prun.steps)
    return fitness, stats

def do_selection(prun, pop_fit, pop):
    if len(pop) < prun.min_pop:
        return individual_generator(provide_id())
    
    return weighted_selection_one(pop, pop_fit, individual_generator, new_fitness=0.5, id_=provide_id())[0]

def do_mutation(prun, sol):
    sol.mutate()
    return sol


def pick_last_exported(exporter):
    generations = list(filter(SM_GEN.match, exporter.list()))
    if len(generations) == 0:
        return None, 0
    
    last = max(generations, key=lambda x: int(SM_GEN.extract(x).get("generation")))
    b_gen = int(SM_GEN.extract(last).get("generation")) + 1
    exporter.print(f"Found generation {b_gen - 1}", "reload")
    pop = exporter.load(last)
    return pop, b_gen


def main(prun):
    prun.history = dict()
    exporter = Exporter(name=prun.name, copy_stdout=True)
    definition = """
    
    """
    exporter.print(definition, slot="definition")
    best = 0
    
    # setup
    pop, batch_gen = pick_last_exported(exporter)
    
    if pop is None:
        sol = do_init(prun)
        pop = [sol]
        batch_gen = 0
    else:
        sol = pop[-1]
        
    for i in range(batch_gen * prun.batch_size,
                   prun.n_gen * prun.batch_size):
        fit, stats = do_fitness(prun, sol)
        sol.fit, sol.stats = fit, stats
        
        if i % 100 == 0:
            exporter.print(f"Step {i}")
        if fit > best:
            exporter.print(f"++ Best {fit} for generation {i}")
            best = fit
            
        monitor = sol
        prun.history[i] = monitor
        # exporter(monitor, f"generation_g{generation}")
        
        sub_pop = pop[-prun.max_pop:]
        sol = do_selection(prun, [s.fit for s in sub_pop], sub_pop)
            
        sol = do_mutation(prun, sol)
        pop.append(sol)
        
        if (i + 1) % 100 == 0:
            batch_gen = (i + 1) // 100
            exporter(pop[-prun.max_pop:], SM_GEN.fill(generation=batch_gen))
        
    return best


class ParamRun(O):
    pop_size = 50
    batch_size = 100
    n_gen = 100
    current_gen = 0
    end_time = 89
    ref = REF
    name = None
    min_pop = 20
    max_pop = 50


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
        pargs.name = f"mutationv3_t{pargs.threshold}_n{pargs.number}_o{pargs.norm}_i{pargs.id}"
    
    NB_GENES = pargs.number
    VALUE_MUTATION = pargs.mutations
    
    args = ParamRun()
    args.n_gen = pargs.generations
    args.norm = pargs.norm
    
    class ObjectiveStep(O):
        start_time = 0
        end_time = 0
        max_fitness = 4
        min_fitness = pargs.threshold
        name = ""
        _passed = False

        def reset(self):
            self._passed = False

        def passed(self):
            if self._passed:
                return
            # print(f"Step {self.name} passed !")
            self._passed = True
    
    example_steps = [
        ObjectiveStep(name="1", start_time=50, end_time=53, score_func=score_bb_size),
        ObjectiveStep(name="2", start_time=53, end_time=56, score_func=score_bb_size),
        ObjectiveStep(name="3", start_time=56, end_time=59, score_func=score_bb_size),
        ObjectiveStep(name="4", start_time=59, end_time=62, score_func=score_bb_size),
        ObjectiveStep(name="5", start_time=62, end_time=65, score_func=score_bb_size),
        ObjectiveStep(name="6", start_time=65, end_time=68, score_func=score_bb_size),
        ObjectiveStep(name="7", start_time=68, end_time=71, score_func=score_bb_size),
        ObjectiveStep(name="8", start_time=71, end_time=74, score_func=score_bb_size),
        ObjectiveStep(name="9", start_time=74, end_time=77, score_func=score_bb_size),
    ]
    
    args.steps = example_steps
    args.name = pargs.name
    main(args)