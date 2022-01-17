from enum import Enum

from lib.sde.grn import GRNMain
from lib.sde.cell import Cell
from model import Action, AbstractCell
from jf.autocompute.jf import O


class GRNCell(Enum):
    Cell = 1


class CellGRN1(AbstractCell):
    def __init__(self, T, start=False, brain=None, index=None, submodel=None, cell_program=None,
                 **kwargs):
        super().__init__(self, T, **kwargs)
        # references
        self.brain = brain
        self.submodel = submodel

        # cell data
        self.appear_time = T
        self.index = index

        self.cell_program: Cell = cell_program if cell_program is not None else \
            self.submodel.cell_cls(self.submodel.grn)

        self.daughter_cells = None
        self.daughter_provide_index = 0

    def tick(self, abs_time, rel_time, neighbours=None):
        self.cell_program.run_step(ts=rel_time)
        return self._read_cell_program()

    def type(self):
        return GRNCell.Cell

    # HERE we read the output of the cell !
    def _read_cell_program(self):
        """
        Read actions such as divide and differentiate.
        Priority order is :
        1) Differentiation
        2) Division
        """
        thr_division = 1
        thr_differentiation = 1
        id_division = 1
        id_differentiation = 2

        if self.cell_program.check_action(id_differentiation, thr_differentiation):
            return Action.DiffNeuron

        elif self.cell_program.check_action(id_division, thr_division):
            self.cell_program.reset(id_division)
            return Action.Divide

        return Action.NoOp

    def __gt__(self, C):
        return self.division_time > C.division_time

    def generate_daughter_cell(self, T, index, tissue_id=None, **kwargs):
        if self.daughter_cells is None:
            self.daughter_cells = self.cell_program.divide()
        if self.daughter_provide_index >= len(self.daughter_cells):
            raise RuntimeError("Trying to take the n{} cell but length is {}".format(
                self.daughter_provide_index, len(self.daughter_cells)
            ))

        daughter_cell_program = self.daughter_cells[self.daughter_provide_index]
        self.daughter_provide_index += 1
        return super().generate_daughter_cell(T, index, tissue_id=tissue_id, cell_program=daughter_cell_program,
                                              **kwargs)


class GRNModelFactory:
    def __init__(self,
                 grn=None,
                 **kwargs
                 ):
        if grn is None:
            grn = GRNMain(**kwargs)

        self.model = O(grn=grn, cell_cls=Cell)

    def generate(self, *args, **kwargs):
        return CellGRN1(*args, **kwargs, submodel=self.model)
