from enum import Enum
from lib.biodata import Tc_
from model import Action, Submodels, AbstractCell
import numpy as np
from scipy.interpolate import interp1d
from numpy.random import choice
from lib.utils import highest_lower, variate_prop

SIG_TC = 12.5


class CellTypeClassic(Enum):
    RG = 0
    IP = 1
    PostMitotic = 2
    Dead = 3

def tc_func_bistate(time_, type_):
    return max(np.random.normal(Tc_(time_), SIG_TC), 10) / 24. # because we are in Days

timesteps = np.array([49, 61, 72, 83, 95])
# values are value to self renew, over value to differentiate
diff_values_RG = np.array([0.73, 0.63, 0.53, 0.43, 0.43])
lin_diff_RG = interp1d(timesteps, diff_values_RG)

diff_values_IP = np.array([0.23, 0.23, 0.23, 0.23, 0.23])
lin_diff_IP = interp1d(timesteps, diff_values_IP)

def diff_func_bistate(time_, type_):
    if type_ == CellTypeClassic.RG:
        val = lin_diff_RG(time_)
        return choice([CellTypeClassic.RG, CellTypeClassic.IP], 1, p=[val, 1-val])[0]
    
    elif type_ == CellTypeClassic.IP:
        val = lin_diff_IP(time_)
        return choice([CellTypeClassic.IP, CellTypeClassic.PostMitotic], 1, p=[val, 1-val])[0]
    

class CellBiStateBasic(AbstractCell):
    """
    Tc will require 2 functions, one for RG and one for IP
    diff function will output p_RG, p_IP for RG
    and p_IP, p_N for IP.
    
    Tuning this model will require to have the proportion of Tbr2+ and Tbr2-
    in the progenitor population.
    """
    def __init__(self, T, start=False, brain=None, index=None, parent_type=None, submodel=None,
                **kwargs):
        super().__init__(self, T, **kwargs)
        # references
        self.brain = brain
        self.submodel = submodel
        
        # cell data
        self.appear_time = T
        self.division_time = np.Inf
        self.index = index
        
        if start:
            self._type = CellTypeClassic.RG
        else:
            self._type = self.submodel.diff(T, parent_type)
        
        if self._type in [CellTypeClassic.RG, CellTypeClassic.IP]:
            self.cycle(T, start)
            
    def tick(self, abs_time, rel_time, neighbours=None):
        if self._type == CellTypeClassic.PostMitotic:
            return Action.DiffNeuron
        
        elif abs_time > self.division_time:
            return Action.Divide
        
        else:
            return Action.NoOp
            
    def cycle(self, T, start=False):
        self.Tc = self.submodel.Tc(T, self.type())
        if start:
            self.eff_Tc = np.random.uniform(0, self.Tc)
        else:
            self.eff_Tc = self.Tc
        self.division_time = T + self.eff_Tc
    
    def __gt__(self, C):
        return self.division_time > C.division_time
    
    def ___generate_daughter_cell(self, T):
        return CellBiStateBasic(T, start=False, brain=self.brain, parent_type=self.type(),
                                submodel=self.submodel)    


class BiStateModelFactory:
    def __init__(self,
                 timesteps=[49, 61, 72, 83, 95],
                 tc_coeff_RG=[1., 1., 1., 1., 1.],
                 tc_coeff_IP=[1., 1., 1., 1., 1.],
                 diff_coeff_RG=[1., 1., 1., 1., 1.],
                 diff_coeff_IP=[1., 1., 1., 1., 1.],
                 diff_values_RG=[0.73, 0.63, 0.53, 0.43, 0.43],
                 diff_values_IP=[0.23, 0.23, 0.23, 0.23, 0.23],
                 **kwargs
                ):
        # DIFF
        # values are value to self renew, over value to differentiate
        lin_diff_RG_ = interp1d(timesteps, diff_values_RG)
        lin_diff_IP_ = interp1d(timesteps, diff_values_IP)
        lin_diff_RG = lambda x: variate_prop(lin_diff_RG_(x), diff_coeff_RG[highest_lower(timesteps, x)])
        lin_diff_IP = lambda x: variate_prop(lin_diff_IP_(x), diff_coeff_IP[highest_lower(timesteps, x)])
        
        # TC
        Tc_IP = lambda x: Tc_(x) * tc_coeff_IP[highest_lower(timesteps, x)]
        Tc_RG = lambda x: Tc_(x) * tc_coeff_RG[highest_lower(timesteps, x)]
            
        def tc_func_bistate(time_, type_):
            if type_ == CellTypeClassic.RG:
                return max(np.random.normal(Tc_RG(time_), SIG_TC), 10) / 24. # because we are in Days
            
            elif type_ == CellTypeClassic.IP:
                return max(np.random.normal(Tc_IP(time_), SIG_TC), 10) / 24. # because we are in Days

        def diff_func_bistate(time_, type_):
            if type_ == CellTypeClassic.RG:
                val = lin_diff_RG(time_)
                return choice([CellTypeClassic.RG, CellTypeClassic.IP], 1, p=[val, 1-val])[0]

            elif type_ == CellTypeClassic.IP:
                val = lin_diff_IP(time_)
                return choice([CellTypeClassic.IP, CellTypeClassic.PostMitotic], 1, p=[val, 1-val])[0]

        self.model = Submodels(tc_func_bistate, diff_func_bistate)
        
    def generate(self, *args, **kwargs):
        return CellBiStateBasic(*args, **kwargs, submodel=self.model)