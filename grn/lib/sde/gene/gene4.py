import numpy as np
from lib.sde.util.formula import formula_from_dict
from lib.sde.util.initializer import generate_random_init, generate_random_noise, generate_random_expr2, \
    generate_random_deg2, generate_random_theta, generate_random_m, generate_random_b, \
    generate_random_tree, generate_random_asym


class MetaParamGene4(type):
    values = ["init", "noise", "b", "m", "expr", "deg", "theta", "asym"]

    def __contains__(self, item):
        return item in self.values

    def __iter__(self):
        for i in self.values:
            yield i

    def __getitem__(self, index):
        return self.values[index]


class ParamGene4(metaclass=MetaParamGene4):
    init = 0
    noise = 1
    b = 2
    m = 3
    expr = 4
    deg = 5
    theta = 6
    asym = 7

    generate_func = [
        generate_random_init,
        generate_random_noise,
        generate_random_b,
        generate_random_m,
        generate_random_expr2,
        generate_random_deg2,
        generate_random_theta,
        generate_random_asym,
    ]

    @staticmethod
    def get(name):
        return getattr(ParamGene4, name)

    @staticmethod
    def length():
        return 8


class GeneMain4:
    def __init__(self, idx, nb_genes, nb_membrane_gene, params, init=True, tree=None,
                 activated: bool = True, generate_funcs: dict = None
                 ):
        """
        :param params: a 8xN array
        """
        self.idx = idx
        self.name = f"G_{self.idx}"
        self.nb_genes = nb_genes
        self.nb_membrane_gene = nb_membrane_gene

        self.params = params
        self.var = dict()

        if init:
            for var in ParamGene4:
                if generate_funcs is not None and var in generate_funcs:
                    generate_func = generate_funcs[var]
                else:
                    generate_func = ParamGene4.generate_func[ParamGene4.get(var)]
                setattr(self, var, generate_func())

        if tree is not None:
            self.tree = tree

        else:
            self.tree = generate_random_tree(self.get_all_labels(), max_depth=2, prob_gene=0.5)

        self.activated = activated

    def re_init(self, generate_funcs: dict = None):
        for var in ParamGene4:
            if generate_funcs is not None and var in generate_funcs:
                generate_func = generate_funcs[var]
            else:
                generate_func = ParamGene4.generate_func[ParamGene4.get(var)]
            setattr(self, var, generate_func())
        self.tree = generate_random_tree(self.get_all_labels(), max_depth=2, prob_gene=0.5)

    def get_all_labels(self):
        return set(range(self.nb_genes + self.nb_membrane_gene))

    def get_labels_not_in_tree(self):
        return self.get_all_labels() - self.tree.labels_set()

    def __setattr__(self, key, value):
        if key in ParamGene4:
            self.params[ParamGene4.get(key), self.idx] = value
        else:
            object.__setattr__(self, key, value)

    def __getattr__(self, item):
        if item in ParamGene4:
            return self.params[ParamGene4.get(item), self.idx]
        else:
            raise AttributeError(item)

    def init_quant(self):
        return np.random.gamma(self.init, 1)

    def set_tree(self, tree):
        self.tree = tree

    def set_params(self, params):
        self.params = params

    def print_formula(self):
        print(repr(self.tree))

    def toggle_activation(self):
        self.activated = not self.activated

    def compute_expression(self, activations):
        return self.tree(activations)

    def set_tree_dict(self, tree_dict):
        self.tree = formula_from_dict(tree_dict)

    def tree_as_dict(self):
        return self.tree.to_dict()

    def get_info(self, cell=None):
        dict_info = {
            var: getattr(self, var) for var in ParamGene4
        }

        if cell is not None:
            dict_info.update(dict(
                activation=cell.activation[self.idx],
                expression=cell.expression[self.idx],
                derivative=cell.derivative[self.idx],
                quantities=cell.quantities[self.idx],
            ))

        return dict_info

    def __repr__(self):
        string = f">> {self.name}: " \
            + "".join([
                "{}: {:.2f}; ".format(var, getattr(self, var))
                for var in ParamGene4
            ]) \
            + f"tree : {repr(self.tree)}"
        return string
