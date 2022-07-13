#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".13"


# In[2]:


import pandas as pd
from brain import BrainModel
from submodels import factories
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from random import shuffle
import random
import argparse

from lib.score import (
    fate_corr, score_both_size_norm, shrink_and_align_stats, score_stats_norm
)
from lib.preprocess import *
from lib.callback import (
    cell_number_callback, progenitor_number_callback, neuron_number_callback,
    TargetPopulation, TagNumberCallback,
)
from lib.sde.grn.grn5 import GRNMain5 as GRNMain
from lib.sde.mutate import (
    mutate_grn5 as mutate_grn,
    mutate_gene5 as mutate_gene,
    mutate_tree_2 as mutate_tree
)

from lib.ga.utils import weighted_selection_one, normalize_fitness_values
from lib.utils import pick_best, pick_last
from jf.utils.export import Exporter
from jf.autocompute.jf import O, L
from itertools import product
import jf.models.stringmodel as sm
from lib.analyser import show_curve, show_curve_progenitor
from jf.models.stringmodel import read_model
from lib.utils import normalize_time, align_time


_count = -1
def provide_id():
    global _count
    _count += 1
    return _count


REF = O(
    stats=pd.read_csv("reference/ref_tristate2.csv"),  # ref is a mean
)
SM_GEN = read_model("generation")
_MUTATE_FUNC = None


def individual_generator(id_=-1, cb_init=None):
    sol = Solution(GRNMain(5, 0, 1, generate_funcs=cb_init), id_=id_)
    sol.grn.set_mutable()
    sol.grn.genes[0].init = 1
    for gene in sol.grn.genes:
        gene.noise = max(1, gene.noise)
    sol.grn.compile()
    return sol


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
        _MUTATE_FUNC(self.grn)


def score_bb_size(bb, ref, *args, **kwargs):
    stats = bb.stats.copy()
    stats, ref_stats = align_time(stats, ref.stats)
    stats_p, ref_p = normalize_time(stats, ref_stats, "progenitor_pop_size", "progenitor_pop_size")
    stats_n, ref_n = normalize_time(stats, ref_stats, "neuron_pop_size", "neuron_pop_size",
                                    "progenitor_pop_size", "progenitor_pop_size")
    last_time_stats, last_time_ref = max(stats.time), max(ref_stats.time)
    
    ref_p = ref_p.set_index("time")
    ref_n = ref_n.set_index("time")
    stats_p = stats_p.set_index("time")
    stats_n = stats_n.set_index("time")
    
    prog = stats_p.loc[last_time_stats]["progenitor_pop_size"]
    neuron = stats_n.loc[last_time_stats]["neuron_pop_size"]
    
    ref_prog = ref_p.loc[last_time_ref]["progenitor_pop_size"]
    ref_neuron = ref_n.loc[last_time_ref]["neuron_pop_size"]

    return 1 / max(1, 1000 - abs(prog - ref_prog) - abs(neuron - ref_neuron))


def get_bb(prun, grn):
    ccls = factories["grn5"](grn=grn)
    callbacks = dict(
        progenitor_pop_size=progenitor_number_callback,
        whole_pop_size=cell_number_callback,
        neuron_pop_size=neuron_number_callback,
    )
    bb = BrainModel(time_step=0.25, verbose=False, start_population=prun.size, max_pop_size=5e2,
            cell_cls=ccls, end_time=prun.end_time, start_time=56, silent=True, opti=True,
              run_tissue=True, monitor_callbacks=callbacks, tag_func=None)
    return bb


def fitness_multistep(prun, grn, steps):
    total_fitness = 0
    stop = False
    previous_time = None
    bb = get_bb(prun, grn)
    # first step
    for step in steps:
        if not bb.run_until(step.end_time):
            stop = True
        
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


def do_init(prun):
    return individual_generator(provide_id(), prun.cb_init)

def do_fitness(prun, sol):
    fitness, stats = fitness_multistep(prun, sol.grn, prun.steps)
    return fitness, stats

def do_selection(prun, pop_fit, pop):
    if len(pop) < prun.min_pop:
        return individual_generator(provide_id(), prun.cb_init)
    
    pop_fit = normalize_fitness_values(pop_fit)
    
    return weighted_selection_one(pop, pop_fit, lambda x: individual_generator(x, prun.cb_init), new_fitness=0.05, id_=provide_id())[0]

def do_mutation(prun, sol):
    sol.mutate()
    return sol


class ObjectiveStep(O):
    start_time = 0
    end_time = 0
    max_fitness = 1e9
    min_fitness = 1
    name = ""
    _passed = False
    
    def reset(self):
        self._passed = False
    
    def passed(self):
        if self._passed:
            return
        print(f"Step {self.name} passed !")
        self._passed = True
    
example_steps = [
    ObjectiveStep(name="3", start_time=56, end_time=86, score_func=score_bb_size, min_fitness=0.2),
]

class ParamRun(O):
    pop_size = 50
    batch_size = 50
    n_gen = 6
    current_gen = 0
    end_time = 86
    ref = REF
    min_pop = 50
    max_pop = 50

def get_prun(size=5, exponent=1):
    prun = ParamRun()
    prun.cb_init = dict()
    prun.size = size
    prun.exponent = exponent
    prun.steps = example_steps
    return prun


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
        
        if (i + 1) % prun.batch_size == 0:
            exporter.print("Saving ...")
            batch_gen = (i + 1) // 100
            exporter(pop[-prun.max_pop:], SM_GEN.fill(generation=batch_gen))
        
    return best


callback_init = dict(
    init=lambda: np.random.beta(1.5, 3) * 3,
    b=lambda: np.random.beta(1.5, 3) * 5,
    expr=lambda: 1,
    deg=lambda: 0.1,
    noise=lambda: np.random.beta(1.5, 3) * 1,
    asym=lambda: 5,
)

def mutate_grn_sparse(grn, temperature=0.1, sparsity=1.0):
    grn.set_mutable()
    shape = grn._params.shape
    r = random.random()
    param_prob = 0.8
    if r < param_prob:
        mask = (np.random.uniform(0, 1, shape) < sparsity)
        coeff = np.random.normal(0, temperature, shape)
        true_coeff = mask * coeff + 1
        grn._params *= true_coeff
    else:
        one_gene = random.choice(grn.genes)
        one_gene.tree = mutate_tree(one_gene.tree, one_gene.get_labels_not_in_tree())
    grn.compile()
    
def mutate_grn_ctrl(grn):
    grn.set_mutable()
    one_gene = random.choice(grn.genes)
    mutate_gene(one_gene)
    grn.compile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", type=int, default=5,
                        help="Number of genes")
    parser.add_argument("-t", "--temperature", type=float, default=0.1,
                        help="Temperature of mutation")
    parser.add_argument("-x", "--sparsity", type=float, default=0.0,
                        help="Proportion, if 0.0 then 1 single mutation in the matrix")
    parser.add_argument("--name", type=str, required=False, default=None,
                        help="Name to be saved")
    pargs = parser.parse_args()

    prun = get_prun()
    prun.cb_init = callback_init
    prun.name = pargs.name
    prun.size = pargs.size
    if pargs.sparsity == 0:
        _MUTATE_FUNC = mutate_grn_ctrl
    else:
        _MUTATE_FUNC = lambda grn: mutate_grn_sparse(grn, temperature=pargs.temperature,
                                                     sparsity=pargs.sparsity)
    res = main(prun)
    
    final_res = [fitness_multistep(prun, res.grn, prun.steps)[0]
                for i in range(5)]
    exporter(final_res, "result")

    




