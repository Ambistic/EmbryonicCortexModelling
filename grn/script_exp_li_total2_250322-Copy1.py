#!/usr/bin/env python
# coding: utf-8

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".13"
import pandas as pd
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
from lib.sde.grn.grn5 import GRNMain5 as GRNMain
from lib.sde.mutate import mutate_grn5 as mutate_grn, SparseMutator
from lib.ga.utils import weighted_selection_one, normalize_fitness_values
from lib.utils import pick_best, pick_last
from jf.utils.export import Exporter
from jf.autocompute.jf import O, L
import jf.models.stringmodel as sm
from jf.models.stringmodel import read_model
from lib.utils import normalize_time, align_time
from jf.utils.helper import provide_id
from lib.bank.scores import fitness_multistep, ObjectiveStep, score_linobj
from lib.bank.params import BaseParam, callback_init
from lib.bank.generators import individual_generator, Solution, set_mutate_func

SM_GEN = read_model("generation")


def do_init(prun):
    return individual_generator(prun, provide_id())

def do_fitness(prun, sol):
    fitness, stats = fitness_multistep(prun, sol.grn, prun.steps)
    return fitness, stats

def do_selection(prun, pop_fit, pop):
    if len(pop) < prun.min_pop:
        return individual_generator(prun, provide_id())
    
    pop_fit = normalize_fitness_values(pop_fit)
    
    return weighted_selection_one(pop, pop_fit, lambda x: individual_generator(prun, x),
                                  new_fitness=prun.new_fitness, id_=provide_id())[0]

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
    exporter = Exporter(name="exp_li_total2/" + prun.name, copy_stdout=True)
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
            batch_gen = (i + 1) // prun.batch_size
            exporter(pop[-prun.batch_size:], SM_GEN.fill(generation=batch_gen))
        
    return best

"""
“normal”, “fixed noise” (init and mutation), “no noise” (init and mutation), “no asym” (param to 100), “no intercell
"""

def parametrize(args, type="normal"):
    if type == "fixednoise":
        args.cb_init["noise"] = lambda: 1
        def hook(grn):
            for gene in grn.genes:
                gene.noise = 1
        set_mutate_func(SparseMutator(hook=hook))
    
    elif type == "nonoise":
        args.cb_init["noise"] = lambda: 0
        def hook(grn):
            for gene in grn.genes:
                gene.noise = 0
        set_mutate_func(SparseMutator(hook=hook))
    
    elif type == "noasym":
        args.cb_init["asym"] = lambda: 1000
        def hook(grn):
            for gene in grn.genes:
                gene.asym = 1000
        set_mutate_func(SparseMutator(hook=hook))
    
    elif type == "nointercell":
        args.n_intergenes = 0
        
    elif type == "onlyintercell":  # noise = 0
        args.cb_init["noise"] = lambda: 0
        args.cb_init["asym"] = lambda: 1000
        def hook(grn):
            for gene in grn.genes:
                gene.asym = 1000
                gene.noise = 0
        set_mutate_func(SparseMutator(hook=hook))
    
    elif type == "onlyasym":
        args.n_intergenes = 0
        args.cb_init["noise"] = lambda: 0
        def hook(grn):
            for gene in grn.genes:
                gene.noise = 0
        set_mutate_func(SparseMutator(hook=hook))
    
    elif type == "onlynoise":
        args.n_intergenes = 0
        args.cb_init["asym"] = lambda: 1000
        def hook(grn):
            for gene in grn.genes:
                gene.asym = 1000
        set_mutate_func(SparseMutator(hook=hook))
    
    elif type == "nothing":
        args.n_intergenes = 0
        args.cb_init["noise"] = lambda: 0
        args.cb_init["asym"] = lambda: 1000
        def hook(grn):
            for gene in grn.genes:
                gene.asym = 1000
                gene.noise = 0
        set_mutate_func(SparseMutator(hook=hook))
        
    elif type == "normal":
        pass
    
    else:
        raise ValueError("Unknown type")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str, default="normal",
                        help="Type of model")
    parser.add_argument("--name", type=str, required=False, default="test170322",
                        help="Name to be saved")
    pargs = parser.parse_args()

    prun = BaseParam(n_gen=10)
    prun.cb_init = callback_init
    prun.name = pargs.name
    prun.steps = [
        ObjectiveStep(name="0", start_time=56, end_time=86, score_func=score_linobj, min_fitness=0.2),
    ]
    parametrize(prun, pargs.type)
    res = main(prun)
    
    final_res = [fitness_multistep(prun, res.grn, prun.steps)
                for i in range(5)]
    exporter(final_res, "result")

    




