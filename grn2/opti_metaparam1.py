#!/usr/bin/env python
# coding: utf-8
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".13"
import pandas as pd
from brain import BrainModel
from submodels import factories
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from random import shuffle
import argparse

from lib.score import (
    fate_corr, score_both_size_norm, shrink_and_align_stats, score_stats_norm
)
from lib.preprocess import *
from lib.callback import (
    cell_number_callback, progenitor_number_callback, neuron_number_callback,
    TargetPopulation, TagNumberCallback, TypeNumberCallback,
)
from lib.sde.grn.grn import GRNMain
from lib.sde.mutate import SparseMutator

from lib.ga.utils import weighted_selection_one, normalize_fitness_values
from lib.utils import pick_best, pick_last
from jf.utils.export import Exporter
from jf.autocompute.jf import O, L
from itertools import product
import jf.models.stringmodel as sm
from lib.analyser import show_curve, show_curve_progenitor
from jf.models.stringmodel import read_model
from lib.utils import normalize_time, align_time
from jf.utils.helper import provide_id

from lib.bank.scores import ObjectiveStep, score_infra_supra as score_bb_size


REF = O(
    stats=pd.read_csv("reference/ref_tristate2.csv"),  # ref is a mean
)
SM_GEN = read_model("generation")
_MUTATE_FUNC = SparseMutator()
_ADVANCEMENT = 0


# In[4]:


def individual_generator(prun, id_=-1, cb_init=None):
    sol = Solution(GRNMain(prun.n_genes, prun.n_regulators, 1, generate_funcs=cb_init), id_=id_)
    sol.grn.set_mutable()
    sol.grn.genes[0].init = 1
    for gene in sol.grn.genes:
        gene.noise = max(1, gene.noise)
    sol.grn.compile()
    return sol


# In[5]:


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


def hook_event_handler(cell_program):
    # prob **2 and **4
    q = np.array(cell_program.quantities)
    if random() < (q[0] / 2)**4:  # missing a reset
        cell_program.quantities = cell_program.quantities.at[0].set(0)
        return Action.Divide, GRNCell.Progenitor
    
    if random() < (q[1] / 2)**1.5 * (1 - q[2] / 2)**1.5:
        return Action.DiffNeuron, GRNCell.PostMitoticInfra
    
    if random() < (q[2] / 2)**1.5 * (1 - q[1] / 2)**1.5:
        return Action.DiffNeuron, GRNCell.PostMitoticSupra
    
    return Action.NoOp, None


# In[8]:


def get_bb(prun, grn):
    from submodels.grn_auto_v1 import GRNCell
    ccls = factories["grn_auto1"](grn=grn, hooks=dict(event_handler=hook_event_handler))
    callbacks = dict(
        progenitor_pop_size=progenitor_number_callback,
        whole_pop_size=cell_number_callback,
        neuron_pop_size=neuron_number_callback,
        infra_pop_size=TypeNumberCallback(GRNCell.PostMitoticInfra, TargetPopulation.whole),
        supra_pop_size=TypeNumberCallback(GRNCell.PostMitoticSupra, TargetPopulation.whole),
    )
    bb = BrainModel(time_step=0.25, verbose=False, start_population=prun.size, max_pop_size=5e2,
            cell_cls=ccls, end_time=prun.end_time, start_time=50, silent=True, opti=True,
              run_tissue=True, monitor_callbacks=callbacks)
    return bb


def fitness_step(prun, grn, step):
    bb = get_bb(prun, grn)
    bb.run_until(step.end_time)
    score_step = step.score_func(bb, prun.ref, max_step=step.end_time, min_step=step.start_time)
    fitness = 1.0 / score_step
        
    return fitness, bb.stats


# In[11]:


def do_init(prun):
    return individual_generator(prun, provide_id(), prun.cb_init)


def do_fitness(prun, sol):
    fitness, stats = fitness_step(prun, sol.grn, prun.step)
    return fitness, stats


def do_selection(prun, pop_fit, pop):
    if len(pop) < prun.min_pop:
        return individual_generator(prun, provide_id(), prun.cb_init)
    
    pop_fit = normalize_fitness_values(pop_fit)
    
    return weighted_selection_one(pop, pop_fit, lambda x: individual_generator(prun, x, prun.cb_init), new_fitness=10., id_=provide_id())[0]


def do_mutation(prun, sol):
    sol.mutate()
    return sol


def main(prun):
    global _ADVANCEMENT
    prun.history = dict()
    exporter = Exporter(name=prun.name, copy_stdout=True)
    best = 0
    
    sol = do_init(prun)
    pop = [sol]
    batch_gen = 0
        
    for i in range(batch_gen * prun.batch_size,
                   prun.n_gen * prun.batch_size):
        _ADVANCEMENT = i / (prun.n_gen * prun.batch_size)
        fit, stats = do_fitness(prun, sol)
        sol.fit, sol.stats = fit, stats
        
        if fit > best:
            best = fit
            
        monitor = sol
        prun.history[i] = monitor
        
        sub_pop = pop[-prun.max_pop:]
        sol = do_selection(prun, [s.fit for s in sub_pop], sub_pop)
            
        sol = do_mutation(prun, sol)
        pop.append(sol)
            
    exporter(pop, prun.run_name)
        
    return best


def build_strategy(prun):
    global _MUTATE_FUNC
    if prun.strategy == "temp_param":
        _MUTATE_FUNC = SparseMutator(temperature_param=prun.strategy_value)
    
    elif prun.strategy == "sparse_param":
        _MUTATE_FUNC = SparseMutator(sparsity_param=prun.strategy_value)
    
    elif prun.strategy == "temp_gene":
        _MUTATE_FUNC = SparseMutator(temperature_tree=prun.strategy_value)
    
    elif prun.strategy == "sparse_gene":
        _MUTATE_FUNC = SparseMutator(sparsity_tree=prun.strategy_value)
    
    elif prun.strategy == "decay_temp_param":
        def mutation_func(grn):
            global _ADVANCEMENT
            func = SparseMutator(temperature_param=prun.strategy_value * (1 - (_ADVANCEMENT / 2)))
            func(grn)
        _MUTATE_FUNC = mutation_func
    
    elif prun.strategy == "decay_temp_gene":
        def mutation_func(grn):
            global _ADVANCEMENT
            func = SparseMutator(temperature_tree=prun.strategy_value * (1 - (_ADVANCEMENT / 2)))
            func(grn)
        _MUTATE_FUNC = mutation_func
    
    elif prun.strategy == "number_gene":
        prun.n_genes = 9
        prun.n_regulators = 7
    
    return prun


class ParamRun(O):
    pop_size = 50
    batch_size = 50
    n_gen = 20
    current_gen = 0
    end_time = 86
    ref = REF
    min_pop = 50
    max_pop = 50
    n_genes = 7
    n_regulators = 5

def get_prun(size=7, exponent=1):
    prun = ParamRun()
    prun.cb_init = dict()
    prun.size = size
    prun.exponent = exponent
    prun.step = ObjectiveStep(name="3", start_time=56, end_time=86, score_func=score_bb_size, min_fitness=0.2)
    return prun


callback_init = dict(
    init=lambda: np.random.beta(1.5, 3) * 3,
    b=lambda: np.random.beta(1.5, 3) * 5,
    expr=lambda: 1,
    deg=lambda: 0.1,
    noise=lambda: np.random.beta(1.5, 3) * 1,
    asym=lambda: 5,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Name of the test")
    parser.add_argument("-v", "--value", type=float)
    parser.add_argument("-i", "--id", type=int)
    args = parser.parse_args()
    
    prun = get_prun()
    prun.cb_init = callback_init
    prun.name = "opti_metaparam_1"
    prun.strategy = args.name
    prun.strategy_value = args.value
    prun.run_id = args.id
    prun.run_name = f"{prun.strategy}_{prun.strategy_value}_{prun.run_id}"
    res = main(prun)
    