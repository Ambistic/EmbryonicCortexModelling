import json
import pickle

import numpy as np

from lib.sde.util.helper import NpEncoder
from lib.sde.util.sde import step_euler_sde, noise, j_step_euler_sde
from jax import jit

try:
    import jax.numpy as jnp
except ImportError:
    jnp = np
    print("Could not load jax, numpy will be used by default")

from lib.sde.grn.grn import GRNBase
from lib.sde.util.serializer import grn_from_json
from lib.sde.gene.gene2 import ParamGene, GeneMain2
from lib.sde.util.func import j_gene_expression


def compute_all_derivatives(genes, expression):
    return jnp.array([gene.compute_derivative(expression) for gene in genes])


class GRNMain2(GRNBase):
    def __init__(self, nb_genes, nb_off_trees, nb_start_gene):
        self.nb_start_gene = nb_start_gene
        self.nb_off_trees = nb_off_trees
        self.nb_genes = nb_genes
        self._compiled = False

        # self._init_quant = init_quant
        # self._nb_mandatory_genes = len(self._init_quant)
        # assert self._nb_mandatory_genes <= self.nb_genes, "Inconsistent number of genes"

        self._params = np.zeros((ParamGene.length(), self.nb_genes))

        # init round
        self.genes = [GeneMain2(idx=i, nb_genes=self.nb_genes, params=self._params)
                      for i in range(self.nb_genes)]

        self.compile()

    def __getattr__(self, item):
        if item in ParamGene:
            return self._params[ParamGene.get(item)]
        else:
            raise AttributeError(f"{item} is not found in ParamGene")

    def compile(self):
        # optimization round
        self._params = jnp.array(self._params)
        for gene in self.genes:
            gene.set_params(self._params)

        self._compiled = True

    def set_mutable(self):
        self._params = np.array(self._params)
        for gene in self.genes:
            gene.set_params(self._params)

        self._compiled = False

    def set_param(self, trees=None, params=None, **kwargs):
        allowed_arr_attr = {
            "init_quant", "b", "m", "expr", "deg", "thr",
            "theta", "trees"
        }

        for k, v in kwargs.items():
            if k in allowed_arr_attr:
                getattr(self, '_' + k, jnp.array(v))

        if params is not None:
            self.set_mutable()
            self._params[...] = np.array(params)

        if trees is not None:
            for tree, gene in zip(trees, self.genes):
                gene.set_tree_dict(tree)

    def get_vector_quantities(self):
        quantities = jnp.array([gene.init_quant() for gene in self.genes])
        # quantities[:len(self._init_quant)] = self._init_quant
        return quantities

    @jit
    def compute_expression(self, quantities):
        return j_gene_expression(quantities, self._params[ParamGene.b],
                                 self._params[ParamGene.theta], self._params[ParamGene.m])

    def compute_derivative(self, expression):
        return compute_all_derivatives(self.genes, expression)

    @jit
    def compute_quantities(self, quantities, derivative, ts=0.1):
        return jnp.array(j_step_euler_sde(quantities, derivative, self._params[ParamGene.noise], ts))

    def run_step(self, quantities, expression, derivative, ts=0.1):
        """
        :param ts: time step
        :param quantities: quantity of each gene
        :param expression: expression vector
        :param derivative: derivative vector
        """
        assert self._compiled
        # compute the effective expression value
        new_expression = j_gene_expression(quantities, self._params[ParamGene.b],
                                           self._params[ParamGene.theta], self._params[ParamGene.m])

        # compute the derivative (with the tree)
        new_derivative = compute_all_derivatives(self.genes, new_expression)
        # run with sde for one step
        new_quantities = jnp.array(step_euler_sde(quantities, new_derivative, noise, ts))
        return new_quantities, new_expression, new_derivative

    def run_multiple_steps(self, quantities, expression, derivative, dt=1, steps=10):
        for i in range(steps):
            quantities, expression, derivative = self.run_step(
                quantities, expression, derivative, dt / steps
            )
        return quantities, expression, derivative

    def to_json(self):
        json_dict = dict(
            object_name=self.__class__.__name__,
            nb_start_gene=self.nb_start_gene,
            nb_off_trees=self.nb_off_trees,
            nb_genes=self.nb_genes,

            params=list(self._params),

            trees=[gene.tree_as_dict() for gene in self.genes]
        )
        return json.dumps(json_dict, cls=NpEncoder)

    def copy(self):
        json_str = self.to_json()
        new = grn_from_json(json_str)
        new.compile()
        return new

    def __repr__(self):
        return "\n".join([repr(gene) for gene in self.genes])

    def __getstate__(self):
        return pickle.dumps(self.to_json())

    def __setstate__(self, state):
        json_str = pickle.loads(state)
        new = grn_from_json(json_str)
        new.compile()
        self.__dict__ = new.__dict__
