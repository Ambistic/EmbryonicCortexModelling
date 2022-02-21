import numpy as np

from lib.sde.grn.grn import GRNMain


class Cell:
    def __init__(self, grn: GRNMain, quantities=None):
        self.grn = grn
        if quantities is None:
            self.quantities = self.grn.get_vector_quantities()
        else:
            self.quantities = quantities

        self.expression = np.zeros(self.quantities.shape)
        self.derivative = np.zeros(self.quantities.shape)

    def run_step(self, ts=0.1):
        self.grn.run_step(quantities=self.quantities,
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

        return Cell(self.grn, quantities=q1), Cell(self.grn, quantities=q2)

    def check_action(self, id_gene, thr_gene):
        """
        Simple threshold check for a given gene
        More complicated triggers (such as combination of conditions)
        are not yet implemented.
        """
        return self.quantities[id_gene] > thr_gene

    def reset(self, id_gene):
        self.quantities[id_gene] = 0
