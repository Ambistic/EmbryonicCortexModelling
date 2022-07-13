import numpy as np
from lib.sde.formula.helper import formula_from_dict
from lib.sde.util.initializer import generate_random_init, generate_random_noise, generate_random_expr2, \
    generate_random_deg2, generate_random_theta, generate_random_m, generate_random_b, \
    generate_random_tree, generate_random_asym


class MetaParamGene(type):
    values = ["b", "m", "theta", "init", "noise", "expr", "deg", "asym"]

    def __contains__(self, item):
        return item in self.values

    def __iter__(self):
        for i in self.values:
            yield i

    def __getitem__(self, index):
        return self.values[index]


class ParamGene(metaclass=MetaParamGene):
    b = 0
    m = 1
    theta = 2
    init = 3
    noise = 4
    expr = 5
    deg = 6
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
        return getattr(ParamGene, name)

    @staticmethod
    def length():
        return 8

    @staticmethod
    def is_expression_param(item):
        return item in ParamGene.values[3:]


class GeneMain:
    def __init__(self, idx, nb_genes, nb_regulators, nb_membrane_gene, params, init=True, tree=None,
                 generate_funcs: dict = None
                 ):
        """
        :param params: a 8xN array
        """
        self.idx = idx
        self.name = f"G_{self.idx}"
        self.nb_genes = nb_genes
        self.nb_regulators = nb_regulators
        self.nb_membrane_gene = nb_membrane_gene

        self.params = params
        self.var = dict()

        if init:
            for var in ParamGene:
                if generate_funcs is not None and var in generate_funcs:
                    generate_func = generate_funcs[var]
                else:
                    generate_func = ParamGene.generate_func[ParamGene.get(var)]
                setattr(self, var, generate_func())

        if tree is not None:
            self.tree = tree

        else:
            self.tree = generate_random_tree(self.get_all_labels(), mode="and")

    def re_init(self, generate_funcs: dict = None):
        for var in ParamGene:
            if generate_funcs is not None and var in generate_funcs:
                generate_func = generate_funcs[var]
            else:
                generate_func = ParamGene.generate_func[ParamGene.get(var)]
            setattr(self, var, generate_func())
        self.tree = generate_random_tree(self.get_all_labels(), mode="and")

    def get_all_labels(self):
        return set(range(self.nb_regulators)) | {"0z"}

    def get_labels_not_in_tree(self):
        return self.get_all_labels() - self.tree.labels_set()

    def tree_not_full(self):
        return len(self.get_labels_not_in_tree()) >= 1

    def __setattr__(self, key, value):
        if key in ParamGene:
            self.params[ParamGene.get(key), self.idx] = value
        else:
            object.__setattr__(self, key, value)

    def __getattr__(self, item):
        if item in ParamGene:
            return self.params[ParamGene.get(item), self.idx]
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
            var: getattr(self, var) for var in ParamGene
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
        string = f"G> {self.name}: " \
            + "".join([
                "{}={:.2f}; ".format(var, getattr(self, var))
                for var in ParamGene
            ]) \
            + f"tree={repr(self.tree)}"
        return string
