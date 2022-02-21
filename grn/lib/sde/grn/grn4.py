import json
import pickle

import numpy as np

from lib.sde.util.helper import NpEncoder
from lib.sde.util.sde import noise, j_step_euler_sde, step_euler_sde3
from jax import jit

try:
    import jax.numpy as jnp
except ImportError:
    jnp = np
    print("Could not load jax, numpy will be used by default")

from lib.sde.grn.grn import GRNBase
from lib.sde.util.serializer import grn_from_json
from lib.sde.gene.gene4 import GeneMain4, ParamGene4
from lib.sde.util.func import j_gene_activation, j_gene_derivation


def compute_all_expressions(genes, activation):
    return jnp.array([gene.compute_expression(activation) for gene in genes])


class GRNMain4(GRNBase):
    def __init__(self, nb_genes, nb_off_trees, nb_membrane_gene,
                 generate_funcs: dict = None):
        self.nb_membrane_gene = nb_membrane_gene
        self.nb_off_trees = nb_off_trees
        self.nb_genes = nb_genes
        self._compiled = False
        self.var = dict()

        self._params = np.zeros((ParamGene4.length(), self.nb_genes))

        # init round
        self.genes = [GeneMain4(idx=i, nb_genes=self.nb_genes,
                                nb_membrane_gene=self.nb_membrane_gene, params=self._params,
                                generate_funcs=generate_funcs)
                      for i in range(self.nb_genes)]
        
        self.id_membrane_genes = np.array(list(range(self.nb_genes - self.nb_membrane_gene, self.nb_genes)))

        self.compile()

    def __getattr__(self, item):
        if item in ParamGene4:
            return self._params[ParamGene4.get(item)]
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

    def set_param(self, trees=None, params=None, var=None, **kwargs):
        if params is not None:
            self.set_mutable()
            self._params[...] = np.array(params)

        if trees is not None:
            for tree, gene in zip(trees, self.genes):
                gene.set_tree_dict(tree)

        if var is not None:
            self.var = var

    def get_vector_quantities(self):
        quantities = jnp.array([gene.init_quant() for gene in self.genes])
        return quantities

    def get_id_membrane_genes(self):
        return self.id_membrane_genes

    @jit
    def compute_activation(self, quantities):
        return j_gene_activation(quantities, self.b,
                                 self.theta, self.m)

    def compute_expression(self, activation):
        return compute_all_expressions(self.genes, activation)

    @jit
    def compute_derivative(self, expression, quantity):
        return j_gene_derivation(expression, quantity, self.expr,
                                 self.deg)

    @jit
    def compute_quantities(self, quantities, derivative, ts=0.1):
        return jnp.array(j_step_euler_sde(quantities, derivative, self.noise, ts))

    def run_step(self, quantities, activation, expression, environment, derivative, ts=0.1):
        """
        :param ts: time step
        :param quantities: quantity of each gene
        :param activation: activation vector
        :param expression: expression vector
        :param derivative: derivative vector
        """
        assert self._compiled
        # compute the effective expression value
        new_activation = j_gene_activation(quantities, self.b,
                                           self.theta, self.m)

        full_activation = jnp.concatenate([new_activation, environment])

        # compute the expression (with the tree)
        new_expression = compute_all_expressions(self.genes, full_activation)

        # compute the derivative
        new_derivative = j_gene_derivation(new_expression, quantities, self.expr,
                                           self.deg)
        # run with sde for one step
        new_quantities = jnp.array(step_euler_sde3(quantities, new_derivative, noise, ts,
                                                   self.noise))
        return new_quantities, new_activation, new_expression, new_derivative

    def run_multiple_steps(self, quantities, activation, expression, derivative, dt=1, steps=10):
        for i in range(steps):
            quantities, activation, expression, derivative = self.run_step(
                quantities, activation, expression, derivative, dt / steps
            )
        return quantities, activation, expression, derivative

    def to_json(self):
        json_dict = dict(
            object_name=self.__class__.__name__,
            nb_membrane_gene=self.nb_membrane_gene,
            nb_off_trees=self.nb_off_trees,
            nb_genes=self.nb_genes,
            var=self.var,

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
