from enum import Enum
from lib.population import CellPopulation


class TargetPopulation(Enum):
    progenitor = 0
    postmitotic = 1
    whole = 2


def progenitor_number_callback(cp: CellPopulation):
    return len(cp.tissue_population)


def neuron_number_callback(cp: CellPopulation):
    return len(cp.post_mitotic)


def cell_number_callback(cp: CellPopulation):
    return len(cp.tissue_population) + len(cp.post_mitotic)


class TypeNumberCallback:
    """This class assumes that the type must be inside the progenitor pool"""
    def __init__(self, typename):
        self.typename = typename

    def __call__(self, cp: CellPopulation):
        return len([x for x in cp.tissue_population.values()
                    if cp.base_population[x].type() == self.typename])


class TagNumberCallback:
    """This class handles progenitor, post mitotic and whole pool"""
    def __init__(self, target_population, tagname, tagvalue, neg=False):
        assert target_population in TargetPopulation
        self.target_population = target_population
        self.tagname = tagname
        self.tagvalue = tagvalue
        self.neg = neg

    def __call__(self, cp: CellPopulation):
        if self.target_population is TargetPopulation.progenitor:
            ref_indexes = cp.tissue_population.values()

        elif self.target_population is TargetPopulation.postmitotic:
            # TODO not ok
            ref_indexes = cp.post_mitotic  # is this cells or ids ?

        elif self.target_population is TargetPopulation.whole:
            ref_indexes = cp.base_population.keys()
        else:
            raise ValueError("`target_population` not understood")

        if self.neg:
            return len([x for x in ref_indexes
                        if cp.base_population[x].tag.get(self.tagname) != self.tagvalue])

        else:
            return len([x for x in ref_indexes
                        if cp.base_population[x].tag.get(self.tagname) == self.tagvalue])
