from lib.sde.grn.grn import GRNMain
from lib.sde.formula.operator import And, Or
from lib.sde.formula import Var, build_var
import random
import numpy as np


_op_2 = [And, Or]


def p(c):
    """Probability function for a constant `c`"""
    return random.random() < c


def mutate_var(grn):
    """Multiply by 0.9 or 1.1 with probability of 1 / 20"""
    if "temperature_param" in grn.var and p(1/20):
        grn.var["temperature_param"] *= 0.9 if p(0.5) else 1.1
        
    if "sparsity_param" in grn.var and p(1/20):
        grn.var["sparsity_param"] *= 0.9 if p(0.5) else 1.1
        
    if "temperature_tree" in grn.var and p(1/20):
        grn.var["temperature_tree"] *= 0.9 if p(0.5) else 1.1
        
    if "sparsity_tree" in grn.var and p(1/20):
        grn.var["sparsity_tree"] *= 0.9 if p(0.5) else 1.1


def singular_shift(value, temperature, sparsity):
    if p(sparsity):
        return value + np.random.normal(0, temperature)
    return value


def mutate_param(grn, temperature, sparsity):
    shape = grn._params.shape
    mask = (np.random.uniform(0, 1, shape) < sparsity)
    coeff = np.random.normal(0, temperature, shape)
    true_coeff = mask * coeff + 1
    grn._params *= true_coeff


def mutate_trees(grn: GRNMain, temperature, sparsity):
    p_new = 0.1
    for element in grn.genes + grn.regulators:
        for sub_el in element.tree.children:
            sub_el.weight = min(1, singular_shift(sub_el.weight, temperature, sparsity))
            if sub_el.weight < 0:
                element.tree.remove_child(sub_el)

        if p(p_new) and element.tree_not_full():
            child_name = random.choice(tuple(element.get_labels_not_in_tree()))
            sign = random.choice([True, False])
            weight = abs(np.random.normal(0, temperature))
            element.tree.add_child(build_var(name=child_name, sign=sign, weight=weight))


class SparseMutator:
    def __init__(self, temperature_param=0.1, sparsity_param=0.2,
                 temperature_tree=0.1, sparsity_tree=0.2, hook=None):
        self.temperature_param = temperature_param
        self.sparsity_param = sparsity_param
        self.temperature_tree = temperature_tree
        self.sparsity_tree = sparsity_tree
        self.hook = hook
        
    def __call__(self, grn):
        grn.set_mutable()
        mutate_var(grn)
        mutate_param(grn, self.temperature_param, self.sparsity_param)
        mutate_trees(grn, self.temperature_tree, self.sparsity_tree)

        if self.hook is not None:
            self.hook(grn)

        grn.compile()


class EvolvableMutator:
    def __init__(self, temperature_param=0.1, sparsity_param=0.2,
                 temperature_tree=0.1, sparsity_tree=0.2, hook=None):
        self.temperature_param = temperature_param
        self.sparsity_param = sparsity_param
        self.temperature_tree = temperature_tree
        self.sparsity_tree = sparsity_tree
        self.hook = hook
        
    def __call__(self, grn):
        grn.set_mutable()
        mutate_var(grn)
        
        temperature_param = grn.var.get("temperature_param", self.temperature_param)
        sparsity_param = grn.var.get("sparsity_param", self.sparsity_param)
        temperature_tree = grn.var.get("temperature_tree", self.temperature_tree)
        sparsity_tree = grn.var.get("sparsity_tree", self.sparsity_tree)
        
        mutate_param(grn, temperature_param, sparsity_param)
        mutate_trees(grn, temperature_tree, sparsity_tree)

        if self.hook is not None:
            self.hook(grn)

        grn.compile()
        