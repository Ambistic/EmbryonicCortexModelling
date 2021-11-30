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
    val = lin_diff(time_)
    if type_ == CellTypeClassic.RG:
        val = lin_diff_RG(time_)
        return choice([CellTypeClassic.RG, CellTypeClassic.IP], 1, p=[val, 1-val])[0]
    
    elif type_ == CellTypeClassic.IP:
        val = lin_diff_IP(time_)
        return choice([CellTypeClassic.IP, CellTypeClassic.PostMitotic], 1, p=[val, 1-val])[0]
    

class CellBiStateLI(AbstractCell):
    """
    Tc will require 2 functions, one for RG and one for IP
    diff function will output p_RG, p_IP for RG
    and p_IP, p_N for IP.
    
    Tuning this model will require to have the proportion of Tbr2+ and Tbr2-
    in the progenitor population.
    """
    def __init__(self, T, start=False, brain=None, index=None, parent_type=None, submodel=None,
                neighbours=[], **kwargs):
        super().__init__(self, T, **kwargs)
        # references
        self.brain = brain
        self.submodel = submodel
        
        # cell data
        self.appear_time = T
        self.division_time = np.Inf
        self.index = index
        
        if start:
            # self._type = CellTypeClassic.RG
            self._type = self.submodel.start_diff(T, self.brain, self.gpn_id)
        else:
            self._type = self.submodel.diff(T, parent_type, self.brain, self.gpn_id, ngbs=neighbours)
        
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
    
# xtime = np.arange(49, 95, 0.1)
# plot_function(xtime, lin_diff_IP, lin_diff_RG, figure=True)

class BiStateLIModelFactory:
    def __init__(self,
                 timesteps=[49, 61, 72, 83, 95],
                 tc_coeff_RG=[1., 1., 1., 1., 1.],
                 tc_coeff_IP=[1., 1., 1., 1., 1.],
                 bias_ratio=[0.15, 0.2, 0, 0, 0],  # the higher is the bias, the more there are RG
                 smooth=0.0,
                 start_val=0.0,
                 **kwargs
                ):
        # DIFF
        # values are value to self renew, over value to differentiate
        # lin_diff_RG = interp1d(timesteps, [x*y for x, y in zip(diff_values_RG, diff_coeff_RG)])
        # lin_diff_IP = interp1d(timesteps, [x*y for x, y in zip(diff_values_IP, diff_coeff_IP)])
        lin_bias_ratio = interp1d(timesteps, bias_ratio)
        
        # TC
        Tc_IP = lambda x: Tc_(x) * tc_coeff_IP[highest_lower(timesteps, x)]
        Tc_RG = lambda x: Tc_(x) * tc_coeff_RG[highest_lower(timesteps, x)]
        
        def repr_():
            timesteps = np.arange(49, 95, 0.1)
            fig = plt.figure(figsize=(12, 8))
            fig.suptitle("Parameters of submodel 2", fontsize=14)

            plt.subplot(2, 2, 1)
            plt.title("Diff RG")
            plt.ylim(0, 1)
            plt.plot(timesteps, list(map(lin_diff_RG, timesteps)))

            plt.subplot(2, 2, 2)
            plt.title("Diff IP")
            plt.ylim(0, 1)
            plt.plot(timesteps, list(map(lin_diff_IP, timesteps)))

            plt.subplot(2, 2, 3)
            plt.ylim(0, 100)
            plt.title(f"Tc (h) RG, sigma={SIG_TC}, min=10")
            plt.plot(timesteps, list(map(Tc_RG, timesteps)))

            plt.subplot(2, 2, 4)
            plt.ylim(0, 100)
            plt.title(f"Tc (h) IP, sigma={SIG_TC}, min=10")
            plt.plot(timesteps, list(map(Tc_IP, timesteps)))
            
        def tc_func_bistate(time_, type_):
            if type_ == CellTypeClassic.RG:
                return max(np.random.normal(Tc_RG(time_), SIG_TC), 10) / 24. # because we are in Days
            
            elif type_ == CellTypeClassic.IP:
                return max(np.random.normal(Tc_IP(time_), SIG_TC), 10) / 24. # because we are in Days

        # this function that ends in the self.model must return a cell type
        # intended to work in variate prop mode, which is interesting for robustness
        # however, does not explain the original emergence of stochasticity
        def diff_func_bistate(time_, type_, ngbs=[]):
            ratio_ngb = np.mean([0 if x.type() == CellTypeClassic.RG else 1 for x in ngbs] or [0.5])
            C = 2**(2 * ratio_ngb - 1)
            if type_ == CellTypeClassic.RG:
                val = get_val_from_neighborhood(brain)
                val = lin_diff_RG(time_)
                val = variate_prop(val, C)
                return choice([CellTypeClassic.RG, CellTypeClassic.IP], 1, p=[val, 1-val])[0]

            elif type_ == CellTypeClassic.IP:
                val = lin_diff_IP(time_)
                return choice([CellTypeClassic.IP, CellTypeClassic.PostMitotic], 1, p=[val, 1-val])[0]
            
        def diff_func_bistate_LI(time_, type_, brain, gpn_id, ngbs=[]):
            if type_ == CellTypeClassic.RG:
                val = get_val_from_neighborhood(time_, brain, gpn_id)
                return choice([CellTypeClassic.RG, CellTypeClassic.IP], 1, p=[val, 1-val])[0]

            elif type_ == CellTypeClassic.IP:
                val = lin_diff_IP(time_)
                return choice([CellTypeClassic.IP, CellTypeClassic.PostMitotic], 1, p=[val, 1-val])[0]
            
        def get_val_from_neighborhood(time_, brain, gpn_id):
            threshold = 0.5
            ls_types = brain.get_ngbs_types(gpn_id)
            # compute ratio of IP in the neighbours
            if len(ls_types) == 0:
                ratio = 0
            else:
                ratio = np.mean([T == CellTypeClassic.IP for T in ls_types])
                
            if ratio + lin_bias_ratio(time_) < threshold:
            # if ratio < threshold:
                return 0 + smooth  # if not enough IP, then no chance staying a RG
            else:
                return 1 - smooth  # if enough IP, then stays a RG
            
        def diff_start(time_, brain, gpn_id):
            val = 1 - start_val
            return choice([CellTypeClassic.RG, CellTypeClassic.IP], 1, p=[val, 1-val])[0]

        self.model = Submodels(tc_func_bistate, diff_func_bistate_LI, repr_=repr_, start_diff=diff_start)
        
    def generate(self, *args, **kwargs):
        return CellBiStateLI(*args, **kwargs, submodel=self.model)