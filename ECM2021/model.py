import numpy as np
import seaborn as sns
import tqdm
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
from utils import nop
from numpy.random import choice
from scipy.interpolate import splev, splrep, interp1d
from gpn2 import GrowingPlanarNetwork
from collections import Counter
from enum import Enum


START_POPULATION_SQRT = 10
START_TIME = 49
END_TIME = 94
SIG_TC = 12.5


class Action(Enum):
    NoOp = 0
    Divide = 1
    Die = 2
    DiffNeuron = 3
    Migrate = 4


class Submodels:
    """
    This class implements all parameters that are required, also under the callback form
    """

    def __init__(self, Tc_func, diff_func, repr_=None, start_diff=None):
        self.Tc = Tc_func
        self.diff = diff_func
        self.repr_ = repr_
        self.start_diff = start_diff

    def show(self):
        if self.repr_:
            self.repr_()


# Definition of the class
class Brain:

    ##################
    # INITIALISATION #
    ##################

    def __init__(
        self,
        start_population=START_POPULATION_SQRT,
        start_time=START_TIME,
        end_time=END_TIME,
        time_step=1,
        sig_tc=SIG_TC,
        cell_cls=None,
        verbose=False,
        silent=False,
        check=False,
    ):
        # debug args
        self.verbose = verbose
        self.silent = silent
        self.check = check

        # model params
        self.start_population = start_population
        self.start_time = start_time
        self.end_time = end_time
        self.time_step = time_step
        self.sig_tc = SIG_TC
        self.cell_cls = self.build_cell_cls(cell_cls)

        # internal var
        self.id_count = -1
        self.range = np.arange(self.start_time, self.end_time, self.time_step)
        self.lenrange = len(self.range)
        self.current_step = 0

        # init
        self.gpn = GrowingPlanarNetwork()
        self.init_mapping()
        self.initiate_population()
        self.init_stats()
        self.snapshots = list()

    def build_cell_cls(self, cell_cls):
        from submodels import factories
        if callable(cell_cls):
            return cell_cls

        elif isinstance(cell_cls, tuple):
            name, args, kwargs = cell_cls
            factory = factories[name](*args, **kwargs)
            return factory.generate

    def debug(self, x):
        if self.verbose:
            print(x)
            
    def info(self, x):
        if not self.silent:
            print(x)

    def set_reference_model(self, callback):
        self.reference_model_callback = callback
        
    def init_mapping(self):
        self.mapping = {
            Action.Divide: self.divide,
            Action.Die: self.remove_cell,
            Action.DiffNeuron: self.set_as_neuron,
            Action.NoOp: nop,
        }
        
    def set_swap(self):
        self.mapping[Action.Divide] = self.divide_and_swap

    def initiate_population(self):
        self.gpn.init_square(self.start_population)
        self.population = dict()
        self.gpn_population = dict()
        for i in self.gpn.G.nodes:
            index = self.new_cell_id()
            self.population[index] = self.cell_cls(self.start_time,
                    start=True, brain=self,
                    index=index, gpn_id=i)

            self.gpn_population[i] = index

        self.root_population = self.population.copy()
        self.post_mitotic = list()  # we must count in order to easily get the neurons

    ##################
    #### CLOCKING ####
    ##################

    def run(self):
        for T in np.arange(self.start_time, self.end_time, self.time_step):
            if not self._tick(T, self.time_step):
                print("Not enough cells anymore")
                return
            
    def run_one_step(self):
        if self.current_step >= self.lenrange:
            print("Run is over")
            return False
        
        T = self.range[self.current_step]
        self._tick(T, self.time_step)
        self.current_step += 1
        return True

    def _tick(self, absolute_time, relative_time):
        self.info(f"Ticking abs : {absolute_time}, step : {relative_time}")
        self._sanity_check()

        # take care of cells
        for C_id in list(self.gpn_population.values()):
            if len(self.gpn_population) < (self.start_population**2 / 2):
                return False
            C = self.population[C_id]
            neighbours = self.get_neighbours(C)
            action = C.tick(absolute_time, relative_time, neighbours)
            self.run_action(action, C, absolute_time)

        # monitor
        stats = dict(
            progenitor_pop_size=len(self.gpn_population),
            whole_pop_size=len(self.gpn_population) + len(self.post_mitotic),
            time=absolute_time,
            **self.population_size_by_type(),
        )

        self.add_stat_time(absolute_time, stats)
        self.snapshots.append(self.gpn.export())
        return True

    def _sanity_check(self):
        if len(self.population) > 1e5:
            raise RuntimeError("Model stops when population increases above 1E5")

    #################
    ###  ACTIONS  ###
    #################

    def run_action(self, action, cell, T):
        self.mapping[action](cell, T)

    def divide(self, cell, T):
        """
        Warning : the id of one daughter is the same as for the mother
        We will have to modify the gpn code to overcome that
        """
        self.debug("Duplicating " + str(cell.index) + " " + str(cell.gpn_id))
        new_gpn_id = self.gpn.duplicate_node(cell.gpn_id)
        time_ = cell.appear_time + cell.eff_Tc
        # time_ = T
        new_cell_1 = cell.generate_daughter_cell(time_, index=self.new_cell_id(),
                gpn_id=new_gpn_id)
        new_cell_2 = cell.generate_daughter_cell(time_, index=self.new_cell_id(),
                gpn_id=cell.gpn_id)
        if self.check:
            self.gpn.check_all()

        self.register_cell(new_cell_1)
        self.register_cell(new_cell_2)
        
    def divide_and_swap(self, cell, T):
        """
        Warning : the id of one daughter is the same as for the mother
        We will have to modify the gpn code to overcome that
        """
        self.debug("Duplicating " + str(cell.index) + " " + str(cell.gpn_id))
        new_gpn_id = self.gpn.duplicate_node(cell.gpn_id)
        
        # select random neighbour then swap with it
        set_ngb = set(self.gpn.ngb(new_gpn_id)) - {cell.gpn_id}
        pick_ngb_1 = random.choice(set_ngb)
        self.gpn.swap_node(pick_ngb_1, new_gpn_id)
        
        # 2nd
        set_ngb = set(self.gpn.ngb(cell.gpn_id)) - {new_gpn_id}
        pick_ngb_2 = random.choice(set_ngb)
        self.gpn.swap_node(pick_ngb_2, cell.gpn_id)
        
        
        time_ = cell.appear_time + cell.eff_Tc
        ngbs1 = self.get_neighbours_from_gpn_id(new_gpn_id, exclude=[cell.gpn_id])
        ngbs2 = self.get_neighbours_from_gpn_id(cell.gpn_id, exclude=[new_gpn_id])
        
        new_cell_1 = cell.generate_daughter_cell(time_, index=self.new_cell_id(),
                gpn_id=new_gpn_id, neighbours=ngbs1)
        new_cell_2 = cell.generate_daughter_cell(time_, index=self.new_cell_id(),
                gpn_id=cell.gpn_id, neighbours=ngbs2)
        
        if self.check:
            self.gpn.check_all()

        self.register_cell(new_cell_1)
        self.register_cell(new_cell_2)

    def set_as_neuron(self, cell, T):
        self.post_mitotic.append(cell)
        self.remove_cell(cell, T)

    def remove_cell(self, cell, T):
        self.debug("Removing " + str(cell.index) + " " + str(cell.gpn_id))
        self.gpn.remove_node(cell.gpn_id)
        if self.check:
            self.gpn.check_all()

        del self.gpn_population[cell.gpn_id]

    def register_cell(self, cell):
        self.population[cell.index] = cell
        self.gpn_population[cell.gpn_id] = cell.index

    #################
    ###  HELPERS  ###
    #################

    def get_neighbours(self, cell):
        ngbs = self.gpn.ngb(cell.gpn_id)
        try:
            return [self.population[self.gpn_population[i]] for i in ngbs]
        except:
            print("DEBUG")
            print(len(self.population), self.gpn_population, cell.gpn_id, ngbs)
            raise
    
    def get_neighbours_from_gpn_id(self, gpn_id, exclude=[]):
        ngbs = self.gpn.ngb(cell.gpn_id)
        ngbs = list(set(ngbs) - set(exclude))
        return [self.population[self.gpn_population[i]] for i in ngbs]
    
    def get_ngbs_types(self, gpn_id):
        ls_types = []
        ngbs = self.gpn.ngb(gpn_id)
        for ngb in ngbs:
            if ngb in self.gpn_population:
                ls_types.append(self.population[self.gpn_population[ngb]].type())
       
        return ls_types

    def new_cell_id(self):
        self.id_count += 1
        return self.id_count
    
    def cell_from_gpn_id(self, gpn_id):
        return self.population[self.gpn_population[gpn_id]]

    ##################
    ### STATISTICS ###
    ##################

    def init_stats(self):
        self.stats = pd.DataFrame({})
        
    def init_stats_old(self):
        self.stats = pd.DataFrame(
            {
                "progenitor_pop_size": [self.start_population],
                "whole_pop_size": [self.start_population],
                "time": [self.start_time],
                **self.population_size_by_type(in_list=True),
            },
            index=[self.start_time],
        )

    def add_stat_time(self, abs_time, stats: dict):
        self.stats = self.stats.append(pd.Series(stats, name=abs_time))

    def population_size_by_type(self, in_list=False):
        d = dict()
        count = Counter([self.population[index].type() for index
            in self.gpn_population.values()])
        for k, v in count.items():
            vv = [v] if in_list else v
            d["size_type_" + str(k.name)] = vv

        return d

    def show_end_curve(self, df_pop):
        xspace = range(int(self.start_time), int(self.end_time) + 1)
        plt.figure(figsize=(18, 12))

        plt.subplot(2, 2, 1)
        plt.title("Number or progenitors")
        plt.plot(xspace, df_pop["cycling_size"] / 100)
        self.reference_model_callback()  # plt.plot(*sorted_number_cells, 'o', x2, y2)

        plt.subplot(2, 2, 2)
        plt.title("Rate of differentiation")
        y_diff = [self.cell_fate_func(x) for x in xspace]
        plt.plot(xspace, y_diff)

        plt.subplot(2, 2, 3)
        plt.title("Number or total cells")
        plt.plot(xspace, df_pop["total_size"] / 100)

        plt.subplot(2, 2, 4)
        plt.title("Time of cell cycle")
        y_Tc = [self.Tc_func(x) for x in xspace]
        plt.ylim(0, max(100, np.max(y_Tc)))
        plt.plot(xspace, y_Tc)
        plt.show()


class AbstractCell:
    def __init__(self, T, start=False, brain=None, index=None, gpn_id=None,
            parent=None, **kwargs):
        self.brain = brain
        self.submodel = None

        # cell data
        self.appear_time = T
        self.division_time = np.Inf
        self.index = index
        self.gpn_id = gpn_id
        self._type = None
        self.Tc = None
        self.eff_Tc = None

        self.parent = parent  # parent is the index of the parent cell
        self.children = list()
        self.divided_tag = False

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

    def generate_daughter_cell(self, T, index, gpn_id=None, **kwargs):
        if gpn_id is None:
            gpn_id = self.gpn_id

        self.children.append(index)

        return type(self)(T, start=False, brain=self.brain, index=index,
                gpn_id=gpn_id, parent=self.index, submodel=self.submodel,
                parent_type=self.type(), **kwargs)
