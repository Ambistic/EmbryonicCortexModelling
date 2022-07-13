from abc import ABC, abstractmethod


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
