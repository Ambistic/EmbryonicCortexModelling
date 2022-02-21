import json
from abc import ABC, abstractmethod
import numpy as np

from lib.sde.gene.gene import Gene, GeneOpt, GeneMain
from lib.sde.util.sde import step_euler_sde, noise
from lib.sde.util.func import gene_expression
from lib.sde.util.helper import NpEncoder
from lib.sde.util.serializer import grn_from_json


class GRNBase(ABC):
    def __init__(self, nb_genes, nb_off_trees, nb_start_gene):
        self.nb_start_gene = nb_start_gene
        self.nb_off_trees = nb_off_trees
        self.nb_genes = nb_genes

    def print_trees(self):
        for gene in self.genes:
            print(gene.name, ":", gene.tree)

    def print_params(self):
        for g in self.genes:
            print(g.name, ":",
                  f"b:{g.b}, m:{g.m}, theta:{g.theta}, thr:{g.thr}, deg:{g.deg}")

    def print_quant(self):
        for gene in self.genes:
            print(gene.name, ":", gene.quantity, gene.expression)

    @abstractmethod
    def run_step(self, *args, **kwargs):
        pass


class GRNSingular(GRNBase):
    def __init__(self, nb_genes, nb_off_trees, nb_start_gene):
        self.nb_start_gene = nb_start_gene
        self.nb_off_trees = nb_off_trees
        self.nb_genes = nb_genes
        self.init_quant = [1]
        self.mandatory_genes = len(self.init_quant)

        assert self.mandatory_genes <= self.nb_genes, "Inconsistent number of genes"

        self.gene_labels = [f"G{i}" for i in range(self.nb_genes)]

        self.genes = [Gene(name=gene, gene_labels=self.gene_labels)
                      for gene in self.gene_labels]

        for index, quant in enumerate(self.init_quant):
            self.genes[index].set_quantity(quant)
        # self.off_trees = [Gene(name=f"T{i}") for i in range(self.nb_off_trees)]

    def run_step(self, ts=0.1):
        """
        :param ts: time step
        """
        # compute the effective expression value
        for gene in self.genes:
            gene.compute_expression()

        expressions = {gene.name: gene.expression for gene in self.genes}
        # compute the derivation (with the tree)
        for gene in self.genes:
            gene.compute_derivative(expressions)

        # run with sde for one step
        der = np.array([gene.derivative for gene in self.genes])
        val = np.array([gene.quantity for gene in self.genes])
        y = step_euler_sde(val, der, noise, ts)
        for gene, y_ in zip(self.genes, y):
            gene.set_quantity(y_)


class GRNOpt(GRNBase):
    def __init__(self, nb_genes, nb_off_trees, nb_start_gene, init_quant=np.array([1])):
        self.nb_start_gene = nb_start_gene
        self.nb_off_trees = nb_off_trees
        self.nb_genes = nb_genes
        self.init_quant = init_quant
        self.mandatory_genes = len(self.init_quant)

        assert self.mandatory_genes <= self.nb_genes, "Inconsistent number of genes"

        self._b = np.zeros(self.nb_genes)
        self._m = np.zeros(self.nb_genes)
        self._expr = np.zeros(self.nb_genes)
        self._deg = np.zeros(self.nb_genes)
        self._thr = np.zeros(self.nb_genes)
        self._theta = np.zeros(self.nb_genes)

        self._expression = np.zeros(self.nb_genes)
        self._quantities = np.zeros(self.nb_genes)
        self._derivative = np.zeros(self.nb_genes)

        self.genes = [GeneOpt(idx=i, nb_genes=self.nb_genes, expr=self._expr, deg=self._deg, thr=self._thr,
                              b=self._b, m=self._m, theta=self._theta,
                              expression=self._expression, quantities=self._quantities,
                              derivative=self._derivative)
                      for i in range(self.nb_genes)]

        self._quantities[:len(self.init_quant)] = self.init_quant

    def set_param(self,
                  init_quant=None,
                  b=None,
                  m=None,
                  expr=None,
                  deg=None,
                  thr=None,
                  theta=None,
                  trees=None,
                  **kwargs,
                  ):
        if init_quant is not None:
            self.init_quant[...] = np.array(init_quant)

        if b is not None:
            self._b[...] = np.array(b)

        if m is not None:
            self._m[...] = np.array(m)

        if expr is not None:
            self._expr[...] = np.array(expr)

        if thr is not None:
            self._thr[...] = np.array(thr)

        if theta is not None:
            self._theta[...] = np.array(theta)

        if deg is not None:
            self._deg[...] = np.array(deg)

        if trees is not None:
            for tree, gene in zip(trees, self.genes):
                gene.set_tree_dict(tree)

    def get_vector_quantities(self):
        quantities = np.zeros(self.nb_genes)
        quantities[:len(self.init_quant)] = self.init_quant
        return quantities

    def compute_expression(self):
        self._expression[...] = gene_expression(self._quantities, self._b, self._theta, self._m)

    def run_step(self, ts=0.1):
        """
        :param ts: time step
        """
        # compute the effective expression value
        self.compute_expression()
        # compute the derivation (with the tree)
        for gene in self.genes:
            gene.compute_derivative(self._expression)

        # run with sde for one step
        y = step_euler_sde(self._quantities, self._derivative, noise, ts)
        self._quantities[...] = y

    def to_json(self):
        json_dict = dict(
            object_name=self.__class__.__name__,
            nb_start_gene=self.nb_start_gene,
            nb_off_trees=self.nb_off_trees,
            nb_genes=self.nb_genes,
            init_quant=list(self.init_quant),
            mandatory_genes=len(self.init_quant),

            b=list(self._b),
            m=list(self._m),
            expr=list(self._expr),
            deg=list(self._deg),
            thr=list(self._thr),
            theta=list(self._theta),

            expression=list(self._expression),
            quantities=list(self._quantities),
            derivative=list(self._derivative),

            trees=[gene.tree_as_dict() for gene in self.genes]
        )
        return json.dumps(json_dict, cls=NpEncoder)

    def copy(self):
        json_str = self.to_json()
        return grn_from_json(json_str)


class GRNMain(GRNBase):
    def __init__(self, nb_genes, nb_off_trees, nb_start_gene, init_quant=np.array([1])):
        self.nb_start_gene = nb_start_gene
        self.nb_off_trees = nb_off_trees
        self.nb_genes = nb_genes

        self._init_quant = init_quant
        self._nb_mandatory_genes = len(self._init_quant)
        assert self._nb_mandatory_genes <= self.nb_genes, "Inconsistent number of genes"

        self._b = np.zeros(self.nb_genes)
        self._m = np.zeros(self.nb_genes)
        self._expr = np.zeros(self.nb_genes)
        self._deg = np.zeros(self.nb_genes)
        self._thr = np.zeros(self.nb_genes)
        self._theta = np.zeros(self.nb_genes)

        self.genes = [GeneMain(idx=i, nb_genes=self.nb_genes, expr=self._expr, deg=self._deg, thr=self._thr,
                              b=self._b, m=self._m, theta=self._theta)
                      for i in range(self.nb_genes)]

    def set_param(self, trees=None, **kwargs):
        allowed_arr_attr = {
            "init_quant", "b", "m", "expr", "deg", "thr",
            "theta", "trees"
        }

        for k, v in kwargs.items():
            if k in allowed_arr_attr:
                getattr(self, '_' + k)[...] = np.array(v)

        if trees is not None:
            for tree, gene in zip(trees, self.genes):
                gene.set_tree_dict(tree)

    def get_vector_quantities(self):
        quantities = np.zeros(self.nb_genes)
        quantities[:len(self._init_quant)] = self._init_quant
        return quantities

    def run_step(self, quantities, expression, derivative, ts=0.1):
        """
        :param ts: time step
        :param quantities: quantity of each gene
        :param expression: expression vector
        :param derivative: derivative vector
        """
        # compute the effective expression value
        expression[...] = gene_expression(quantities, self._b, self._theta, self._m)

        # compute the derivation (with the tree)
        derivative[...] = np.array([gene.compute_derivative(expression) for gene in self.genes])
        # run with sde for one step
        y = step_euler_sde(quantities, derivative, noise, ts)
        quantities[...] = y

    def to_json(self):
        json_dict = dict(
            object_name=self.__class__.__name__,
            nb_start_gene=self.nb_start_gene,
            nb_off_trees=self.nb_off_trees,
            nb_genes=self.nb_genes,
            init_quant=list(self._init_quant),
            mandatory_genes=len(self._init_quant),

            b=list(self._b),
            m=list(self._m),
            expr=list(self._expr),
            deg=list(self._deg),
            thr=list(self._thr),
            theta=list(self._theta),

            trees=[gene.tree_as_dict() for gene in self.genes]
        )
        return json.dumps(json_dict, cls=NpEncoder)

    def copy(self):
        json_str = self.to_json()
        return grn_from_json(json_str)

    def __repr__(self):
        return "\n".join([repr(gene) for gene in self.genes])


