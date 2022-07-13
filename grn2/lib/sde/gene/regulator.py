from lib.sde.formula import formula_from_dict
from lib.sde.util.initializer import generate_random_tree


class Regulator:
    def __init__(self, idx, nb_genes, nb_regulators, nb_membrane_gene, tree=None,
                 generate_funcs: dict = None
                 ):
        self.idx = idx
        self.nb_genes = nb_genes
        self.nb_regulators = nb_regulators
        self.nb_membrane_gene = nb_membrane_gene
        self.var = dict()
        if generate_funcs is None:
            generate_funcs = dict()

        if tree is not None:
            self.tree = tree

        else:
            generate_tree = generate_funcs.get("regulator_tree", generate_random_tree)
            self.tree = generate_tree(self.get_gene_labels(), mode="or")

    def get_gene_labels(self):
        return set(range(self.nb_genes + self.nb_membrane_gene)) | {"1z"}

    def get_labels_not_in_tree(self):
        return self.get_gene_labels() - self.tree.labels_set()

    def tree_not_full(self):
        return len(self.get_labels_not_in_tree()) >= 1

    def __repr__(self):
        string = f"R> tree={repr(self.tree)}"
        return string

    def tree_as_dict(self):
        return self.tree.to_dict()

    def set_tree_dict(self, tree_dict):
        self.tree = formula_from_dict(tree_dict)
