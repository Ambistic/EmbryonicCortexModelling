from enum import Enum
from lib.biodata import Tc_
from model import Action, Submodels, AbstractCell
import numpy as np
from scipy.interpolate import interp1d
from numpy.random import choice
from lib.utils import highest_lower

SIG_TC = 12.5
    
class CellTypeGP(Enum):
    RG = 0
    IP = 1
    PostMitotic = 2
    Dead = 3
    GP = 4
    
class CellTriStateLI(AbstractCell):
    """
    Tc will require 2 functions, one for RG and one for IP
    diff function will output p_RG, p_IP for RG
    and p_IP, p_N for IP.
    
    Tuning this model will require to have the proportion of Tbr2+ and Tbr2-
    in the progenitor population.
    """
    def __init__(self, T, start=False, brain=None, index=None, parent_type=None, submodel=None,
                neighbours=[], **kwargs):
        super().__init__(T, **kwargs)
        # references
        self.brain = brain
        self.submodel = submodel
        
        # cell data
        self.appear_time = T
        self.division_time = np.Inf
        self.index = index
        
        if start:
            # self._type = CellTypeClassic.RG
            self._type = self.submodel.start_diff(T, self.brain, self.tissue_id)
        else:
            self._type = self.submodel.diff(T, parent_type, self.brain, self.tissue_id, ngbs=neighbours)
        
        if self._type in [CellTypeGP.RG, CellTypeGP.IP, CellTypeGP.GP]:
            self.cycle(T, start)
            
    def tick(self, abs_time, rel_time, neighbours=None):
        if self._type == CellTypeGP.PostMitotic:
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
        return CellTriStateBasic(T, start=False, brain=self.brain, parent_type=self.type(),
                                submodel=self.submodel)
    
class TriStateLIModelFactory:
    def __init__(self,
                 timesteps=[49, 61, 72, 83, 95],
                 tc_coeff_RG=[1., 1., 1., 1., 1.],
                 tc_coeff_IP=[1., 1., 1., 1., 1.],
                 diff_values_RG_GP=[1., 1., 1., 0.8, 0.6], # prob GP vs IP
                 diff_values_IP=[0.23, 0.23, 0.23, 0.23, 0.23],
                 bias_ratio=[0.2, 0.15, -0.1, -0.1, -0.1],  # the higher is the bias, the more there are RG
                 smooth=0.0,
                 start_val=0.0,
                 GP_as_IP=False,
                 **kwargs
                ):
        # DIFF
        # values are value to self renew, over value to differentiate
        # lin_diff_RG = interp1d(timesteps, [x*y for x, y in zip(diff_values_RG, diff_coeff_RG)])
        # lin_diff_IP = interp1d(timesteps, [x*y for x, y in zip(diff_values_IP, diff_coeff_IP)])
        lin_bias_ratio = interp1d(timesteps, bias_ratio)
        lin_diff_RG_GP = interp1d(timesteps, diff_values_RG_GP)
        lin_diff_IP = interp1d(timesteps, diff_values_IP)
        # lin_diff_RG = lambda x: lin_diff_RG_(x) * diff_coeff_RG[highest_lower(timesteps, x)]
        
        # TC
        Tc_IP = lambda x: Tc_(x) * tc_coeff_IP[highest_lower(timesteps, x)]
        Tc_RG = lambda x: Tc_(x) * tc_coeff_RG[highest_lower(timesteps, x)]
        
        def repr_():
            timesteps = np.arange(49, 95, 0.1)
            fig = plt.figure(figsize=(12, 8))
            fig.suptitle("Parameters of submodel 2", fontsize=14)

            plt.subplot(2, 2, 1)
            plt.title("Diff RG")
            plt.ylim(-0.05, 1.05)
            plt.plot(timesteps, list(map(lin_diff_RG_IP, timesteps)))
            plt.plot(timesteps, list(map(lin_diff_RG_GP, timesteps)))

            plt.subplot(2, 2, 2)
            plt.title("Diff IP")
            plt.ylim(-0.05, 1.05)
            plt.plot(timesteps, list(map(lin_diff_IP, timesteps)))

            plt.subplot(2, 2, 3)
            plt.ylim(-5, 105)
            plt.title(f"Tc (h) RG, sigma={SIG_TC}, min=10")
            plt.plot(timesteps, list(map(Tc_RG, timesteps)))

            plt.subplot(2, 2, 4)
            plt.ylim(-5, 105)
            plt.title(f"Tc (h) IP, sigma={SIG_TC}, min=10")
            plt.plot(timesteps, list(map(Tc_IP, timesteps)))
            
        def tc_func_tristate(time_, type_):
            if type_ == CellTypeGP.RG:
                return max(np.random.normal(Tc_RG(time_), SIG_TC), 10) / 24. # because we are in Days
            
            elif type_ == CellTypeGP.IP:
                return max(np.random.normal(Tc_IP(time_), SIG_TC), 10) / 24. # because we are in Days
            
            elif type_ == CellTypeGP.GP:
                return max(np.random.normal(Tc_RG(time_), SIG_TC), 10) / 24. * 100 # we suppose GP won't divide

        def diff_func_tristate(time_, type_):
            if type_ == CellTypeGP.RG:
                val1 = lin_diff_RG_IP(time_)
                val2 = lin_diff_RG_GP(time_)
                return choice([CellTypeGP.RG, CellTypeGP.IP, CellTypeGP.GP], 1,
                              p=[val1, val2 - val1, 1 - val2])[0]

            elif type_ == CellTypeGP.IP:
                val = lin_diff_IP(time_)
                return choice([CellTypeGP.IP, CellTypeGP.PostMitotic], 1, p=[val, 1-val])[0]
            
            elif type_ == CellTypeGP.GP:
                return CellTypeGP.GP
            
        def diff_func_tristate_LI(time_, type_, brain, tissue_id, ngbs=[]):
            if type_ == CellTypeGP.RG:
                val = get_val_from_neighborhood(time_, brain, tissue_id)
                if np.random.random() < val:
                    return CellTypeGP.RG
                
                else:
                    val2 = lin_diff_RG_GP(time_)
                    return choice([CellTypeGP.GP, CellTypeGP.IP], 1, p=[1 - val2, val2])[0]

            elif type_ == CellTypeGP.IP:
                val = lin_diff_IP(time_)
                return choice([CellTypeGP.IP, CellTypeGP.PostMitotic], 1, p=[val, 1 - val])[0]
            
            elif type_ == CellTypeGP.GP:
                return CellTypeGP.GP
            
        def get_val_from_neighborhood(time_, brain, tissue_id):
            threshold = 0.5
            ls_types = brain.get_ngbs_types(tissue_id)
            # compute ratio of IP in the neighbours
            IP_group = [CellTypeGP.IP, CellTypeGP.GP] if GP_as_IP else [CellTypeGP.IP]
            if len(ls_types) == 0:
                ratio = 0
            else:
                ratio = np.mean([T in IP_group for T in ls_types])
                
            if ratio + lin_bias_ratio(time_) < threshold:
            # if ratio < threshold:
                return 0 + smooth  # if not enough IP, then no chance staying a RG
            else:
                return 1 - smooth  # if enough IP, then stays a RG
            
        def diff_start(time_, brain, tissue_id):
            val = 1 - start_val
            return choice([CellTypeGP.RG, CellTypeGP.IP], 1, p=[val, 1-val])[0]


        self.model = Submodels(tc_func_tristate, diff_func_tristate_LI, repr_=repr_, start_diff=diff_start)
        
    def generate(self, *args, **kwargs):
        return CellTriStateLI(*args, **kwargs, submodel=self.model)
