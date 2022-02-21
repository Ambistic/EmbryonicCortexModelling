from lib.sde.gene.gene import *
from lib.sde.util.formula import formula_from_dict
from lib.sde.util.initializer import generate_random_init, generate_random_noise, generate_random_expr, \
    generate_random_deg, generate_random_thr, generate_random_theta, generate_random_m, generate_random_b, \
    generate_random_tree


class MetaParamGene(type):
    values = ["init", "noise", "b", "m", "expr", "deg", "thr", "theta"]

    def __contains__(self, item):
        return item in self.values

    def __iter__(self):
        for i in self.values:
            yield i
            
    def __getitem__(self, index):
        return self.values[index]


class ParamGene(metaclass=MetaParamGene):
    init = 0
    noise = 1
    b = 2
    m = 3
    expr = 4
    deg = 5
    thr = 6
    theta = 7

    generate_func = [
        generate_random_init,
        generate_random_noise,
        generate_random_b,
        generate_random_m,
        generate_random_expr,
        generate_random_deg,
        generate_random_thr,
        generate_random_theta,
    ]

    @staticmethod
    def get(name):
        return getattr(ParamGene, name)

    @staticmethod
    def length():
        return 8


class GeneMain2:
    def __init__(self, idx, nb_genes, params, init=True, tree=None, activated: bool = True,
                 ):
        """
        :param params: a 8xN array
        """
        self.idx = idx
        self.name = f"G_{self.idx}"
        self.nb_genes = nb_genes

        self.params = params

        if init:
            for var in ParamGene:
                generate_func = ParamGene.generate_func[ParamGene.get(var)]
                setattr(self, var, generate_func())

        if tree is not None:
            self.tree = tree

        else:
            self.tree = generate_random_tree(self.get_all_labels(), max_depth=3, prob_gene=0.5)

        self.activated = activated

    def get_all_labels(self):
        return set(range(self.nb_genes))

    def get_labels_not_in_tree(self):
        return self.get_all_labels() - self.tree.labels_set()

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

    def compute_derivative(self, expressions):
        return self.tree(expressions)

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
                expression=cell.expression[self.idx],
                derivative=cell.derivative[self.idx],
                quantities=cell.quantities[self.idx],
            ))

        return dict_info

    def __repr__(self):
        string = f">> {self.name}: " \
            + "".join([
                "{}: {:.2f}; ".format(var, getattr(self, var))
                for var in ParamGene
            ]) \
            + f"tree : {repr(self.tree)}"
        return string
