import numpy as np
from jax import jit

from lib.sde.util.func import j_gene_activation, j_gene_derivation
from lib.sde.util.sde import j_step_euler_sde
from lib.sde.util.helper import batch

try:
    import jax.numpy as jnp
except ImportError:
    jnp = np

from lib.sde.grn.grn3 import GRNMain3


class Cell3:
    def __init__(self, grn: GRNMain3, quantities=None):
        self.grn = grn
        if quantities is None:
            self.quantities = self.grn.get_vector_quantities()
        else:
            self.quantities = quantities

        self.activation = jnp.zeros(self.quantities.shape)
        self.expression = jnp.zeros(self.quantities.shape)
        self.derivative = jnp.zeros(self.quantities.shape)

    def set_quantities(self, quantities):
        self.quantities = quantities

    def set_expression(self, expression):
        self.expression = expression

    def set_activation(self, activation):
        self.activation = activation

    def set_derivative(self, derivative):
        self.derivative = derivative

    def run_step(self, ts=0.1):
        self.quantities, self.activation, self.expression, self.derivative = self.grn.run_step(
            activation=self.activation,
            quantities=self.quantities,
            derivative=self.derivative,
            expression=self.expression,
            ts=ts)

    def get_gene_info(self, name):
        return self.grn.genes[name].get_info(cell=self)

    def print_gene_info(self, name):
        info = self.get_gene_info(name)
        print(f"===== Printing gene {name} =====")
        for k, v in sorted(info.items()):
            print(f"-> {k} : {v}")

    def divide(self):
        # fixed coef and bias for now
        coef = 10
        bias = 2
        asymmetries = np.array([np.random.beta(bias + coef * q, bias + coef * q)
                               for q in self.quantities])

        q1 = self.quantities * asymmetries * 2
        q2 = self.quantities * (1 - asymmetries) * 2

        return Cell3(self.grn, quantities=jnp.array(q1)), Cell3(self.grn, quantities=jnp.array(q2))

    def check_action(self, id_gene, thr_gene):
        """
        Simple threshold check for a given gene
        More complicated triggers (such as combination of conditions)
        are not yet implemented.
        """
        return self.quantities[id_gene] > thr_gene

    def reset(self, id_gene):
        """
        Reset the gene to 0 value
        """
        self.quantities = self.quantities.at[id_gene].set(0)


@jit
def batch_compute_activation(ls_quantities, b, theta, m):
    return [j_gene_activation(q, b, theta, m) for q in ls_quantities]


def batch_compute_expression(genes, activations):
    gene_funcs = [gene.tree.get_compiled() for gene in genes]
    return j_batch_compute_expressions(gene_funcs, activations)


@jit
def batch_compute_derivatives(ls_expr, ls_q, expr_coeff, deg_coeff):
    return [j_gene_derivation(expr, q, expr_coeff, deg_coeff)
            for expr, q in zip(ls_expr, ls_q)]


# @jit
def j_batch_compute_expressions(gene_funcs, expressions):
    return [
        jnp.array([gene_func(expression) for gene_func in gene_funcs])
        for expression in expressions
    ]


@jit
def batch_compute_quantities(quantities, derivatives, noise_param, ts):
    return [
        jnp.array(j_step_euler_sde(q, der, noise_param, ts))
        for q, der in zip(quantities, derivatives)
    ]


class CellBatch3:
    def __init__(self, cells):
        assert len(cells) > 0, "A CellBatch cannot be empty"
        self.cells = cells
        self.grn = self.cells[0].grn

    def run_step_old(self, ts=0.1):
        quantities = [c.quantities for c in self.cells]
        activations = batch_compute_activation(quantities, self.grn.b, self.grn.theta, self.grn.m)
        expressions = batch_compute_expression(self.grn.genes, activations)
        derivatives = batch_compute_derivatives(expressions, quantities, self.grn.expr, self.grn.deg)
        quantities = batch_compute_quantities(quantities, derivatives, self.grn.noise, ts)

        self.dispatch_quantities(quantities)
        self.dispatch_derivatives(derivatives)
        self.dispatch_expressions(expressions)
        self.dispatch_activations(activations)

    def run_step(self, ts=0.1, batch_size=256):
        for cell_batch in batch(self.cells, batch_size):
            self.run_step_batch(cell_batch, batch_size, ts)

    def run_step_batch(self, cell_batch, size, ts):
        """
        :param size: size of the expected batch in order to pad
        :param cell_batch: batch of cells
        :param ts: time step
        """
        quantities = [c.quantities for c in cell_batch]
        if len(quantities) < size:
            shape = quantities[0].shape
            pad_length = size - len(quantities)
            pad_quantities = [jnp.zeros(shape) for i in range(pad_length)]
            quantities += pad_quantities
        elif len(quantities) > size:
            raise ValueError("Cannot have size lower than batch length : {} size for a batch of length {}".format(
                size, len(quantities)
            ))

        activations = batch_compute_activation(quantities, self.grn.b, self.grn.theta, self.grn.m)
        expressions = batch_compute_expression(self.grn.genes, activations)
        derivatives = batch_compute_derivatives(expressions, quantities, self.grn.expr, self.grn.deg)
        quantities = batch_compute_quantities(quantities, derivatives, self.grn.noise, ts)

        self.dispatch_quantities_batch(quantities, cell_batch)
        self.dispatch_derivatives_batch(derivatives, cell_batch)
        self.dispatch_expressions_batch(expressions, cell_batch)
        self.dispatch_activations_batch(activations, cell_batch)

    def dispatch_quantities(self, quantities):
        for c, q in zip(self.cells, quantities):
            c.set_quantities(q)

    def dispatch_derivatives(self, derivatives):
        for c, der in zip(self.cells, derivatives):
            c.set_derivative(der)

    def dispatch_expressions(self, expressions):
        for c, expr in zip(self.cells, expressions):
            c.set_expression(expr)

    def dispatch_activations(self, activations):
        for c, activ in zip(self.cells, activations):
            c.set_activation(activ)

    @staticmethod
    def dispatch_quantities_batch(quantities, cell_batch):
        for c, q in zip(cell_batch, quantities):
            c.set_quantities(q)

    @staticmethod
    def dispatch_derivatives_batch(derivatives, cell_batch):
        for c, der in zip(cell_batch, derivatives):
            c.set_derivative(der)

    @staticmethod
    def dispatch_expressions_batch(expressions, cell_batch):
        for c, expr in zip(cell_batch, expressions):
            c.set_expression(expr)

    @staticmethod
    def dispatch_activations_batch(activations, cell_batch):
        for c, activ in zip(cell_batch, activations):
            c.set_activation(activ)


