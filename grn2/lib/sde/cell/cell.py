import numpy as np
from jax import jit, random

from lib.sde.util.func import j_gene_activation, j_gene_derivation
from lib.sde.util.sde import j_step_euler_sde_2
from lib.sde.util.helper import batch, finebatch

try:
    import jax.numpy as jnp
except ImportError:
    jnp = np

from lib.sde.grn.grn import GRNMain


def notzero(x):
    return max(0.001, x)


class Cell:
    """Hello"""
    def __init__(self, grn: GRNMain, quantities=None, activation=None, hook_init=None):
        self.brain_environment = []
        self.grn = grn
        if quantities is None:
            quantities = self.grn.get_vector_quantities()
        self.quantities = quantities
            
        if activation is None:
            activation = jnp.zeros(self.quantities.shape)
        self.activation = activation
        
        self.environment = jnp.zeros(len(self.grn.get_id_membrane_genes()))
        self.expression = jnp.zeros(self.quantities.shape)
        self.derivative = jnp.zeros(self.quantities.shape)
        
        if hook_init is not None:
            res = hook_init(self.quantities, self.activation, self.environment,
                            self.expression, self.derivative, self.grn)
            (self.quantities, self.activation, self.environment,
             self.expression, self.derivative) = res

    def integrate_environment(self, cells, hook=None):
        membrane_genes = self.grn.get_id_membrane_genes()
        if not len(cells) or not len(membrane_genes):
            self.environment = jnp.zeros(len(self.grn.get_id_membrane_genes()))
            return
        
        env = np.array([np.array(c.activation)[membrane_genes] for c in cells])
        env = env.sum(axis=0)
        if hook is not None:
            env = hook(env)
        self.environment = jnp.array(env)

    def set_brain_environment(self, cells):
        self.brain_environment = cells

    def set_environment(self, env):
        self.environment = env

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
            environment=self.environment,
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
        asymmetries = np.array([np.random.beta(notzero(a), notzero(a))
                               for a, q in zip(self.grn.asym, self.quantities)])

        q1 = self.quantities * asymmetries * 2
        q2 = self.quantities * (1 - asymmetries) * 2

        return type(self)(self.grn, quantities=jnp.array(q1), activation=self.activation), \
            type(self)(self.grn, quantities=jnp.array(q2), activation=self.activation)

    def check_action(self, id_gene, thr_gene):
        """
        Simple threshold check for a given gene
        More complicated triggers (such as combination of conditions)
        are not yet implemented.
        """
        return np.array(self.quantities)[id_gene] > thr_gene

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
def batch_compute_quantities(quantities, derivatives, noise_param, ts, key):
    shape = (len(quantities), *quantities[0].shape)
    noise_value = random.normal(key, shape)
    return [
        jnp.array(j_step_euler_sde_2(q, der, noise_param, ts, n))
        for q, der, n in zip(quantities, derivatives, noise_value)
    ]


@jit
def batch_concatenate(activations, environment):
    return [jnp.concatenate([act, env]) for act, env in zip(activations, environment)]


@jit
def batch_integrate_environment(environments, membrane_genes):
    """
    if not len(cells):
        self.environment = jnp.zeros(len(self.grn.get_id_membrane_genes()))
        return
    membrane_genes = self.grn.get_id_membrane_genes()
    env = jnp.array([c.activation[membrane_genes] for c in cells])
    env = env.mean(axis=0)  # to stay in [0, 1] interval
    """
    return []


class CellBatch:
    def __init__(self, cells):
        assert len(cells) > 0, "A CellBatch cannot be empty"
        self.cells = cells
        self.grn = self.cells[0].grn
        seed = np.random.randint(4294967295)
        self.key = random.PRNGKey(seed)

    def get_random_key(self):
        self.key, key = random.split(self.key)
        return key

    def run_step_old(self, ts=0.1, batch_size=256):
        for cell_batch in batch(self.cells, batch_size):
            self.run_step_batch(cell_batch, batch_size, ts)

    def run_step(self, ts=0.1, base_size=16, mul=4):
        for cell_batch in finebatch(self.cells, base_size, mul):
            batch_size = max(base_size, len(cell_batch))
            self.run_step_batch(cell_batch, batch_size, ts)

    def run_step_batch(self, cell_batch, size, ts):
        """
        :param size: size of the expected batch in order to pad
        :param cell_batch: batch of cells
        :param ts: time step
        """
        quantities = [c.quantities for c in cell_batch]
        environments = [c.environment for c in cell_batch]
        if len(quantities) < size:
            shape = quantities[0].shape
            pad_length = size - len(quantities)
            pad_quantities = [jnp.zeros(shape) for _ in range(pad_length)]
            quantities += pad_quantities
        elif len(quantities) > size:
            raise ValueError("Cannot have size lower than batch length : {} size for a batch of length {}".format(
                size, len(quantities)
            ))

        all_quantities = batch_concatenate(quantities, environments)
        activations = batch_compute_activation(all_quantities, self.grn.b, self.grn.theta, self.grn.m)
        expressions = batch_compute_expression(self.grn.genes, activations)
        derivatives = batch_compute_derivatives(expressions, quantities, self.grn.expr, self.grn.deg)
        quantities = batch_compute_quantities(quantities, derivatives, self.grn.noise, ts, self.get_random_key())

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


