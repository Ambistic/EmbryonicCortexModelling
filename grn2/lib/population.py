import numpy as np

from lib.utils import nop
from lib.action import Action


class Base:
    verbose = False
    silent = False

    def debug(self, x):
        if self.verbose:
            print(x)

    def info(self, x):
        if not self.silent:
            print(x)


class CellPopulation(Base):
    def __init__(self, brain, ids, cell_cls, start_time):
        self.cell_cls = cell_cls
        self.id_count = -1
        self.brain = brain
        self.base_population = dict()
        self.tissue_population = dict()
        self.mapping = {
            Action.Divide: self.divide,
            Action.Die: self.remove_cell,
            Action.DiffNeuron: self.set_as_neuron,
            Action.NoOp: nop,
        }
        for i in ids:
            index = self.new_cell_id()
            self.base_population[index] = cell_cls(start_time,
                                                   start=True, brain=brain,
                                                   index=index, tissue_id=i)

            self.tissue_population[i] = index

        self.root_population = self.base_population.copy()
        self.post_mitotic = list()  # we must count in order to easily get the neurons

    def cell_from_tissue_id(self, tissue_id):
        return self.base_population[self.tissue_population[tissue_id]]

    def new_cell_id(self):
        self.id_count += 1
        return self.id_count

    def run_action(self, action, cell, T):
        self.mapping[action](cell, T)

    def tick_cell_programs(self, absolute_time, relative_time):
        # take care of cells
        for C_id in list(self.tissue_population.values()):
            C = self.base_population[C_id]
            neighbours = self.brain.get_neighbours(C)
            C.set_neighbourhood(neighbours)
            action = C.tick(absolute_time, relative_time, neighbours)
            self.run_action(action, C, absolute_time)

    def tick_cell_programs_batch(self, absolute_time, relative_time):
        # take care of cells
        cells = [self.base_population[C_id] for C_id in list(self.tissue_population.values())]
        for C in cells:
            neighbours = self.brain.get_neighbours(C)
            C.set_neighbourhood(neighbours)
            # C.lazy_set_neighbourhood(neighbours)

        # this batch function must be provided by the cell class
        self.cell_cls.batch_tick(cells, absolute_time, relative_time)

        for C in cells:
            action = C.get_action()
            self.run_action(action, C, absolute_time)

    def divide(self, cell, T):
        """
        Warning : the id of one daughter is the same as for the mother
        We will have to modify the tissue code to overcome that
        """
        self.debug("Duplicating cell " + str(cell.index) + " located in " + str(cell.tissue_id))
        del self.tissue_population[cell.tissue_id]
        new_tissue_id1, new_tissue_id2 = self.brain.tissue.divide_cell(cell.tissue_id)

        try:
            # TODO clear this part / harmonize
            time_ = cell.appear_time + cell.eff_Tc

        except TypeError:
            time_ = T

        new_cell_1 = cell.generate_daughter_cell(time_, index=self.new_cell_id(),
                                                 tissue_id=new_tissue_id1)
        new_cell_2 = cell.generate_daughter_cell(time_, index=self.new_cell_id(),
                                                 tissue_id=new_tissue_id2)

        self.register_cell(new_cell_1)
        self.register_cell(new_cell_2)

    def set_as_neuron(self, cell, T):
        self.post_mitotic.append(cell)
        self.remove_cell(cell, T)

    def remove_cell(self, cell, T):
        self.debug("Removing " + str(cell.index) + " " + str(cell.tissue_id))
        self.brain.tissue.remove_cell(cell.tissue_id)
        del self.tissue_population[cell.tissue_id]

    def register_cell(self, cell):
        self.base_population[cell.index] = cell
        self.tissue_population[cell.tissue_id] = cell.index


class AbstractCell:
    def __init__(self, T, start=False, brain=None, index=None, tissue_id=None,
                 parent=None, submodel=None, tag=None, **kwargs):
        self.brain = brain
        self.submodel = submodel

        # cell data
        self.appear_time = T
        self.division_time = np.Inf
        self.index = index
        self.tissue_id = tissue_id
        self._type = None
        self.Tc = None
        self.eff_Tc = None

        self.parent = parent  # parent is the index of the parent cell
        self.children = list()
        self.divided_tag = False
        if tag is None:
            tag = dict()
        self.tag = tag

    def get_children(self):
        return [self.brain.population[idx] for idx in self.children]

    def eff_Tc_h(self):
        return self.eff_Tc * 24.

    def Tc_h(self):
        if self.Tc is None:
            return 0.
        return self.Tc * 24.

    def set_index(self, index):
        self.index = index

    def type(self):
        return self._type

    def generate_daughter_cell(self, T, index, tissue_id=None, **kwargs):
        if tissue_id is None:
            tissue_id = self.tissue_id

        self.children.append(index)
        return type(self)(T, start=False, brain=self.brain, index=index,
                          tissue_id=tissue_id, parent=self.index, submodel=self.submodel,
                          parent_type=self.type(), tag=self.tag.copy(), **kwargs)

    def freeze(self):
        return self._type

