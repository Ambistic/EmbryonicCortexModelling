from enum import Enum
from lib.biodata import Tc_
from model import Action, Submodels, AbstractCell
import numpy as np
from scipy.interpolate import interp1d
from numpy.random import choice
from lib.utils import highest_lower, variate_prop

SIG_TC = 12.5
    
class CellTypeMutant(Enum):
    RG = 0
    IP = 1
    PostMitotic = 2
    Dead = 3
    GP = 4
    Mutant = 5
    
class CellTriStateAmbiLIMutant(AbstractCell):
    """
    Tc will require 2 functions, one for RG and one for IP
    diff function will output p_RG, p_IP for RG
    and p_IP, p_N for IP.
    
    Tuning this model will require to have the proportion of Tbr2+ and Tbr2-
    in the progenitor population.
    """
    def __init__(self, T, start=False, parent_type=None,
                neighbours=[], **kwargs):
        super().__init__(T, **kwargs)
        
        if start:
            # self._type = CellTypeClassic.RG
            self._type = self.submodel.start_diff(T, self.brain, self.tissue_id)
        else:
            self._type = self.submodel.diff(T, parent_type, self.brain, self.tissue_id, ngbs=neighbours)
        
        if self._type in [CellTypeMutant.RG, CellTypeMutant.IP, CellTypeMutant.GP, CellTypeMutant.Mutant]:
            self.cycle(T, start)
            
    def tick(self, abs_time, rel_time, neighbours=None):
        if self._type == CellTypeMutant.PostMitotic:
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
    
    
class TriStateAmbiLIMutantModelFactory:
    def __init__(self,
                 timesteps=[49, 61, 72, 83, 95],
                 tc_coeff_RG=[1., 1., 1., 1., 1.],
                 tc_coeff_IP=[1., 1., 1., 1., 1.],
                 diff_values_RG_GP=[1., 1., 1., 0.8, 0.6], # prob GP vs IP knowing it's not renew
                 diff_values_RG_IP=[.7, .6, .4, 0.45, 0.4], # prob IP vs renew
                 diff_values_IP=[0.23, 0.23, 0.23, 0.23, 0.23],
                 bias_ratio=[0.7, 0.62, 0.33, 0.5, 0.5],  # the higher is the bias, the more there are RG
                 coeff_mutant=0.2,
                 smooth=0.0,
                 start_val=0.4,
                 id_mutant=-1,
                 GP_as_IP=False,
                 **kwargs
                ):
        # DIFF
        # values are value to self renew, over value to differentiate
        # lin_diff_RG = interp1d(timesteps, [x*y for x, y in zip(diff_values_RG, diff_coeff_RG)])
        # lin_diff_IP = interp1d(timesteps, [x*y for x, y in zip(diff_values_IP, diff_coeff_IP)])
        lin_bias_ratio = interp1d(timesteps, bias_ratio)
        lin_diff_RG_GP = interp1d(timesteps, diff_values_RG_GP)
        lin_diff_RG_IP = interp1d(timesteps, diff_values_RG_IP)
        lin_diff_IP = interp1d(timesteps, diff_values_IP)
        # lin_diff_RG = lambda x: lin_diff_RG_(x) * diff_coeff_RG[highest_lower(timesteps, x)]
        
        # TC
        Tc_IP = lambda x: Tc_(x) * tc_coeff_IP[highest_lower(timesteps, x)]
        Tc_RG = lambda x: Tc_(x) * tc_coeff_RG[highest_lower(timesteps, x)]
        
        def repr_():
            pass
            
        def tc_func_tristate(time_, type_):
            if type_ == CellTypeMutant.RG:
                return max(np.random.normal(Tc_RG(time_), SIG_TC), 10) / 24. # because we are in Days
            
            elif type_ == CellTypeMutant.Mutant:
                return max(np.random.normal(Tc_RG(time_), SIG_TC), 10) / 24. # because we are in Days
            
            elif type_ == CellTypeMutant.IP:
                return max(np.random.normal(Tc_IP(time_), SIG_TC), 10) / 24. # because we are in Days
            
            elif type_ == CellTypeMutant.GP:
                return max(np.random.normal(Tc_RG(time_), SIG_TC), 10) / 24. * 100 # we suppose GP won't divide

            
        def diff_func_tristate_LI(time_, type_, brain, tissue_id, ngbs=[]):
            if type_ == CellTypeMutant.RG:
                val = get_val_from_neighborhood(time_, brain, tissue_id)
                if np.random.random() < val:
                    return CellTypeMutant.RG
                
                else:
                    val2 = lin_diff_RG_GP(time_)
                    return choice([CellTypeMutant.GP, CellTypeMutant.IP], 1, p=[1 - val2, val2])[0]

            elif type_ == CellTypeMutant.IP:
                val = lin_diff_IP(time_)
                return choice([CellTypeMutant.IP, CellTypeMutant.PostMitotic], 1, p=[val, 1 - val])[0]
            
            elif type_ == CellTypeMutant.GP:
                return CellTypeMutant.GP
            
            elif type_ == CellTypeMutant.Mutant:
                val = get_val_from_neighborhood(time_, brain, tissue_id)
                if np.random.random() < val:
                    return CellTypeMutant.Mutant
                
                else:
                    if np.random.random() < coeff_mutant:  # "second chance"
                        return CellTypeMutant.Mutant
                    val2 = lin_diff_RG_GP(time_)
                    return choice([CellTypeMutant.GP, CellTypeMutant.IP], 1, p=[1 - val2, val2])[0]
            
        def get_val_from_neighborhood(time_, brain, tissue_id):
            threshold = 1
            ls_types = brain.get_ngbs_types(tissue_id)
            # compute ratio of IP in the neighbours
            IP_group = [CellTypeMutant.IP, CellTypeMutant.GP] if GP_as_IP else [CellTypeMutant.IP]
            if len(ls_types) == 0:
                ratio = 0
            else:
                ratio = np.mean([T in IP_group for T in ls_types])
                
            if ratio < threshold - lin_bias_ratio(time_):
                thr_val = 0  # if not enough IP, then no chance staying a RG
            else:
                thr_val = 1  # if enough IP, then stays a RG
                
            rate = lin_diff_RG_IP(time_)
                
            return (1 - smooth) * thr_val + smooth * rate
            
        def diff_start(time_, brain, tissue_id):
            if tissue_id == id_mutant:
                return CellTypeMutant.Mutant
            val = 1 - start_val
            return choice([CellTypeMutant.RG, CellTypeMutant.IP], 1, p=[val, 1-val])[0]


        self.model = Submodels(tc_func_tristate, diff_func_tristate_LI, repr_=repr_, start_diff=diff_start)
        
    def generate(self, *args, **kwargs):
        return CellTriStateAmbiLIMutant(*args, **kwargs, submodel=self.model)
