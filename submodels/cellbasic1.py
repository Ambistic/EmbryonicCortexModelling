from enum import Enum
from biodata import Tc_
from model import Action, Submodels, AbstractCell
import numpy as np
from scipy.interpolate import interp1d
from numpy.random import choice

class CellTypeBasic(Enum):
    Cycling = 0
    PostMitotic = 1
    Dead = 2

SIG_TC = 12.5
    
def tc_func_1(time_, type_):
    return max(np.random.normal(Tc_(time_), SIG_TC), 10) / 24. # because we are in Days

timesteps = np.array([49, 61, 72, 83, 95])
diff_values = np.array([0.73, 0.63, 0.47, 0.45, 0.45])
lin_diff = interp1d(timesteps, diff_values)

def diff_func_1(time_, type_):
    val = lin_diff(time_)
    return choice([CellTypeBasic.Cycling, CellTypeBasic.PostMitotic], 1, p=[val, 1-val])[0]

diff_func_1(49, None)

submodel1 = Submodels(tc_func_1, diff_func_1)

class CellBasic(AbstractCell):
    def __init__(self, T, start=False, brain=None, index=None, gpn_id=None, **kwargs):
        super().__init__(self, T, **kwargs)
        # references
        self.brain = brain
        self.submodel = submodel1
        
        # cell data
        self.appear_time = T
        self.division_time = np.Inf
        self.index = index
        self.gpn_id = gpn_id
        
        if start:
            self._type = CellTypeBasic.Cycling
        else:
            self._type = self.submodel.diff(T, None)
        
        if self._type == CellTypeBasic.Cycling:
            self.cycle(T, start)
            
    def tick(self, abs_time, rel_time, neighbours=None):
        if self._type != CellTypeBasic.Cycling:
            return Action.DiffNeuron
        
        elif abs_time > self.division_time:
            return Action.Divide
        
        else:
            return Action.NoOp
            
    def cycle(self, T, start=False):
        self.Tc = self.submodel.Tc(T, None)
        if start:
            self.eff_Tc = np.random.uniform(0, self.Tc)
        else:
            self.eff_Tc = self.Tc
        self.division_time = T + self.eff_Tc
    
    def __gt__(self, C):
        return self.division_time > C.division_time
    
    def ___generate_daughter_cell(self, T, index=None, gpn_id=None):
        if index is None:
            index = self.index
        if gpn_id is None:
            gpn_id = self.gpn_id
            
        return CellBasic(T, start=False, brain=self.brain)