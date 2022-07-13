from lib.sde.grn.grn import GRNMain as GRNMain
from lib.sde.mutate import mutate_grn5 as mutate_grn
import numpy as np
import random


_MUTATE_FUNC = mutate_grn


def set_mutate_func(func):
    global _MUTATE_FUNC
    _MUTATE_FUNC = func


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


def individual_generator(prun, id_=-1):
    grn = GRNMain(prun.n_genes, 0, prun.n_intergenes, generate_funcs=prun.cb_init)
    
    grn.set_mutable()
    grn.genes[0].init = 1
    for gene in grn.genes:
        gene.noise = max(1, gene.noise)
    grn.compile()
    
    sol = Solution(grn, id_=id_)
    
    return sol


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