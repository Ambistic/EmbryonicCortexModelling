import contextlib
from enum import Enum
import numpy as np

from lib.sde.grn.grn5 import GRNMain5 as GRNMain
from lib.sde.cell.cell5 import Cell5 as Cell, CellBatch5 as CellBatch
from lib.population import AbstractCell
from lib.action import Action
from jf.autocompute.jf import O


class GRNCell(Enum):
    Cell = 1
    Progenitor = 2
    PostMitoticInfra = 3
    PostMitoticSupra = 4


class CellGRNAuto1(AbstractCell):
    def __init__(self, T, start=False, brain=None, index=None, submodel=None, cell_program=None,
                 hooks=None, **kwargs):
        super().__init__(self, T, **kwargs)
        
        if hooks is None:
            hooks = dict()

        self.hooks = hooks
        
        # references
        self.brain = brain
        self.submodel = submodel

        # cell data
        self.appear_time = T
        self.index = index
        self._type = GRNCell.Cell

        self.cell_program: Cell = cell_program if cell_program is not None else \
            self.submodel.cell_cls(self.submodel.grn, hook_init=self.hooks.get("hook_init"))

        self.daughter_cells = None
        self.daughter_provide_index = 0

    def tick(self, abs_time, rel_time, neighbours=None):
        self.cell_program.run_step(ts=rel_time)
        return self._read_cell_program()

    def type(self):
        return self._type

    def set_neighbourhood(self, neighbours):
        # here integrate the signal
        env = [n.cell_program for n in neighbours]
        hook_env_modifier = self.hooks.get("env_modifier")
        self.cell_program.integrate_environment(env, hook_env_modifier)

    def lazy_set_neighbourhood(self, neighbours):
        env = [contextlib.closing for n in neighbours]
        self.cell_program.set_brain_environment(env)

    def get_action(self):
        return self._read_cell_program()

    # HERE we read the output of the cell !
    def _read_cell_program(self):
        if "event_handler" in self.hooks:
            action, _type = self.hooks["event_handler"](self.cell_program)
            if _type is not None:
                self._type = _type
            return action
        """
        Read actions such as divide and differentiate.
        Priority order is :
        1) Differentiation
        2) Division
        """
        thr_division = 2
        thr_differentiation_infra = 2
        thr_differentiation_supra = 2
        id_division = 0  # this is different from version 2
        id_differentiation_infra = 1
        id_differentiation_supra = 2

        if self.cell_program.check_action(id_differentiation_infra, thr_differentiation_infra):
            self._type = GRNCell.PostMitoticInfra
            return Action.DiffNeuron

        elif self.cell_program.check_action(id_differentiation_supra, thr_differentiation_supra):
            self._type = GRNCell.PostMitoticSupra
            return Action.DiffNeuron

        elif self.cell_program.check_action(id_division, thr_division):
            self.cell_program.reset(id_division)
            self._type = GRNCell.Progenitor
            return Action.Divide

        return Action.NoOp

    def __gt__(self, C):
        return self.division_time > C.division_time

    def generate_daughter_cell(self, T, index, tissue_id=None, **kwargs):
        """Requires a provide_index because cell programs are generated simultaneously
        due to asymmetrical sharing of the molecules"""
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

    def freeze(self):
        return dict(
            quantity=np.asarray(self.cell_program.quantities),
            activation=np.asarray(self.cell_program.activation),
            expression=np.asarray(self.cell_program.expression),
            derivative=np.asarray(self.cell_program.derivative),
            environment=np.asarray(self.cell_program.environment),
        )


class GRNModelFactory:
    def __init__(self,
                 grn=None,
                 **kwargs
                 ):
        if grn is None:
            grn = GRNMain(**kwargs)

        self.model = O(grn=grn, cell_cls=Cell)

    def __call__(self, *args, **kwargs):
        return CellGRNAuto1(*args, **kwargs, submodel=self.model)

    @staticmethod
    def batch_tick(cells, absolute_time, relative_time):
        """
        Equivalent of run_step for all cells
        """
        cell_programs = [c.cell_program for c in cells]
        batch = CellBatch(cell_programs)
        batch.run_step(relative_time)
