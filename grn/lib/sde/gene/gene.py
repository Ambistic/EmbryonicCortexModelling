import numpy as np

from lib.sde.util.formula import formula_from_dict
from lib.sde.util.func import gene_expression
from lib.sde.util.initializer import generate_random_expr, generate_random_deg, generate_random_thr, \
    generate_random_theta, generate_random_m, generate_random_b, generate_random_tree


class Gene:
    def __init__(self, name, expr=None, deg=None, thr=None, theta=None,
                 m=None, b=None, tree=None, activated: bool = False,
                 gene_labels=None):
        self.b = b or generate_random_b()
        self.m = m or generate_random_m()
        self.theta = theta or generate_random_theta()
        self.thr = thr or generate_random_thr()
        self.deg = deg or generate_random_deg()
        self.expr = expr or generate_random_expr()
        self.name = name
        self.quantity = 0
        self.expression = 0
        self.derivative = 0
        if tree is not None:
            self.tree = tree

        elif gene_labels is not None:
            self.tree = generate_random_tree(gene_labels, max_depth=3, prob_gene=0.5)

        self.activated = activated

    def set_tree(self, tree):
        self.tree = tree

    def get_expression(self):
        return gene_expression(self.quantity, self.b, self.theta, self.m)

    def print_formula(self):
        print(repr(self.tree))

    def toggle_activation(self):
        self.activated = not self.activated

    def set_quantity(self, q):
        self.quantity = q

    def compute_expression(self):
        self.expression = gene_expression(self.quantity, self.b, self.theta, self.m)

    def compute_derivative(self, expressions):
        self.derivative = self.tree(expressions)

    def add_quantity(self, q):
        self.quantity += q


class GeneOpt:
    def __init__(self, idx, nb_genes, expr, deg, thr, theta,
                 m, b, expression, quantities, derivative,
                 init=True, tree=None, activated: bool = True,
                 ):
        """
        Parameters must be np.arrays

        """
        assert all(map(lambda x: isinstance(x, np.ndarray), (expr, deg, thr, theta, m, b))), "Error in inputs " \
            "parameters shall all bu np.ndarray"
        self.idx = idx
        self.name = f"G_{self.idx}"
        self.nb_genes = nb_genes

        self._b = b
        self._m = m
        self._expr = expr
        self._deg = deg
        self._thr = thr
        self._theta = theta

        self._quantities = quantities
        self._expression = expression
        self._derivative = derivative

        if init:
            self.b = generate_random_b()
            self.m = generate_random_m()
            self.theta = generate_random_theta()
            self.thr = generate_random_thr()
            self.deg = generate_random_deg()
            self.expr = generate_random_expr()

        if tree is not None:
            self.tree = tree

        else:
            self.tree = generate_random_tree(set(range(self.nb_genes)), max_depth=3, prob_gene=0.5)

        self.activated = activated

    @property
    def b(self):
        return self._b[self.idx]

    @b.setter
    def b(self, b):
        self._b[self.idx] = b

    @property
    def m(self):
        return self._m[self.idx]

    @m.setter
    def m(self, m):
        self._m[self.idx] = m

    @property
    def expr(self):
        return self._expr[self.idx]

    @expr.setter
    def expr(self, expr):
        self._expr[self.idx] = expr

    @property
    def thr(self):
        return self._thr[self.idx]

    @thr.setter
    def thr(self, thr):
        self._thr[self.idx] = thr

    @property
    def theta(self):
        return self._theta[self.idx]

    @theta.setter
    def theta(self, theta):
        self._theta[self.idx] = theta

    @property
    def deg(self):
        return self._deg[self.idx]

    @deg.setter
    def deg(self, deg):
        self._deg[self.idx] = deg

    @property
    def expression(self):
        return self._expression[self.idx]

    @expression.setter
    def expression(self, expression):
        self._expression[self.idx] = expression

    @property
    def quantity(self):
        return self._quantities[self.idx]

    @quantity.setter
    def quantity(self, quantity):
        self._quantities[self.idx] = quantity

    @property
    def derivative(self):
        return self._derivative[self.idx]

    @derivative.setter
    def derivative(self, derivative):
        self._derivative[self.idx] = derivative

    def set_tree(self, tree):
        self.tree = tree

    def get_expression(self):
        return gene_expression(self.quantity, self.b, self.theta, self.m)

    def print_formula(self):
        print(repr(self.tree))

    def toggle_activation(self):
        self.activated = not self.activated

    def set_quantity(self, q):
        self.quantity = q

    def compute_expression(self):
        self.expression = gene_expression(self.quantity, self.b, self.theta, self.m)

    def compute_derivative(self, expressions):
        self.derivative = self.tree(expressions)

    def add_quantity(self, q):
        self.quantity += q

    def set_tree_dict(self, tree_dict):
        self.tree = formula_from_dict(tree_dict)

    def tree_as_dict(self):
        return self.tree.to_dict()


class GeneMain:
    def __init__(self, idx, nb_genes, expr, deg, thr, theta,
                 m, b, init=True, tree=None, activated: bool = True,
                 ):
        """
        Parameters must be np.arrays

        """
        assert all(map(lambda x: isinstance(x, np.ndarray), (expr, deg, thr, theta, m, b))), "Error in inputs " \
            "parameters shall all bu np.ndarray"
        self.idx = idx
        self.name = f"G_{self.idx}"
        self.nb_genes = nb_genes

        self._b = b
        self._m = m
        self._expr = expr
        self._deg = deg
        self._thr = thr
        self._theta = theta

        if init:
            self.b = generate_random_b()
            self.m = generate_random_m()
            self.theta = generate_random_theta()
            self.thr = generate_random_thr()
            self.deg = generate_random_deg()
            self.expr = generate_random_expr()

        if tree is not None:
            self.tree = tree

        else:
            self.tree = generate_random_tree(self.get_all_labels(), max_depth=3, prob_gene=0.5)

        self.activated = activated

    def get_all_labels(self):
        return set(range(self.nb_genes))

    def get_labels_not_in_tree(self):
        return self.get_all_labels() - self.tree.labels_set()

    @property
    def b(self):
        return self._b[self.idx]

    @b.setter
    def b(self, b):
        self._b[self.idx] = b

    @property
    def m(self):
        return self._m[self.idx]

    @m.setter
    def m(self, m):
        self._m[self.idx] = m

    @property
    def expr(self):
        return self._expr[self.idx]

    @expr.setter
    def expr(self, expr):
        self._expr[self.idx] = expr

    @property
    def thr(self):
        return self._thr[self.idx]

    @thr.setter
    def thr(self, thr):
        self._thr[self.idx] = thr

    @property
    def theta(self):
        return self._theta[self.idx]

    @theta.setter
    def theta(self, theta):
        self._theta[self.idx] = theta

    @property
    def deg(self):
        return self._deg[self.idx]

    @deg.setter
    def deg(self, deg):
        self._deg[self.idx] = deg

    def set_tree(self, tree):
        self.tree = tree

    def print_formula(self):
        print(repr(self.tree))

    def toggle_activation(self):
        self.activated = not self.activated

    def compute_derivative(self, expressions):
        return self.tree(expressions)

    def set_tree_dict(self, tree_dict):
        self.tree = formula_from_dict(tree_dict)

    def tree_as_dict(self):
        return self.tree.to_dict()

    def get_info(self, cell=None):
        dict_info = dict(
            deg=self.deg,
            theta=self.theta,
            b=self.b,
            m=self.m,
            expr=self.expr,
            thr=self.thr,
            tree=self.tree,
        )

        if cell is not None:
            dict_info.update(dict(
                expression=cell.expression[self.idx],
                derivative=cell.derivative[self.idx],
                quantities=cell.quantities[self.idx],
            ))

        return dict_info

    def __repr__(self):
        string = f">> {self.name}: " \
            + "b: {:.2f}; ".format(self.b) \
            + "m: {:.2f}; ".format(self.m) \
            + "deg: {:.2f}; ".format(self.deg) \
            + "expr: {:.2f}; ".format(self.expr) \
            + "thr: {:.2f}; ".format(self.thr) \
            + "theta: {:.2f}; ".format(self.theta) \
            + f"tree : {repr(self.tree)}"
        return string


