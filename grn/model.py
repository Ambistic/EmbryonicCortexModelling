import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lib.utils import nop
from collections import Counter
from enum import Enum
from functools import lru_cache

from cbmos import CBModelv3
import cbmos.force_functions as ff
import cbmos.solvers.euler_forward as ef


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
    """
    Time steps are defined as follows :
    start_time is the time of init, and init param are logged
    When a step is run from T to T+t, the T is already registered, then
    T+t is run, then T+t is logged.
    Therefore, when we run something until T=N, then the time N
    is computed and logged, then we stop

    """
    opti = False

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
        dt=0.01,
        mu=5.70,
        s=1.0,
        a=5.0,
        rA=1.5,
        opti=False,
        max_pop_size=1e4
    ):
        # debug args
        self.verbose = verbose
        self.silent = silent
        self.opti = opti

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
        self.max_pop_size = max_pop_size

        # init
        self.tissue = CBModelv3(ff.Gls(), ef.solve_ivp, dimension=2,
                        force_args={"mu": mu, "s": s, "rA": rA, "a": a},
                        solver_args={'dt': dt},)
        self.init_mapping()
        self.initiate_population()
        self.init_stats()
        self.snapshots = list()
        self.monitor(self.start_time)

    def build_cell_cls(self, cell_cls):
        from submodels import factories
        if callable(cell_cls):
            return cell_cls

        elif isinstance(cell_cls, tuple):
            name, args, kwargs = cell_cls
            factory = factories[name](*args, **kwargs)
            return factory  # TODO change all generate to __call__

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
        self.tissue.init_square(self.start_population)
        self.population = dict()
        self.tissue_population = dict()
        for i in self.tissue.get_ids():
            index = self.new_cell_id()
            self.population[index] = self.cell_cls(self.start_time,
                    start=True, brain=self,
                    index=index, tissue_id=i)

            self.tissue_population[i] = index

        self.root_population = self.population.copy()
        self.post_mitotic = list()  # we must count in order to easily get the neurons

    ##################
    #### CLOCKING ####
    ##################

    def run(self):
        for T in np.arange(self.start_time, self.end_time, self.time_step):
            if not self._tick(T, self.time_step):
                self.info("Population exploded or extinguished")
                return False
        return True
            
    def run_until(self, time_point):
        """
        Runs until the given time point. Returns True if succeeded and
        returns False is it stopped before
        """
        while (self.current_step < self.lenrange) \
                and (self.range[self.current_step] < time_point):
            T = self.range[self.current_step]
            if not self._tick(T, self.time_step):
                self.info("Population exploded or extinguished")
                return False
            self.current_step += 1

        return True
            
    def run_one_step(self):
        if self.current_step >= self.lenrange:
            print("Run is over")
            return False
        
        T = self.range[self.current_step]
        self._tick(T, self.time_step)
        self.current_step += 1
        return True

    def _tick(self, absolute_time, relative_time):
        self.info(f"Ticking abs : {absolute_time}, step : {relative_time}, size : {len(self.tissue_population)}")
        if not self._sanity_check():
            return False
        
        if len(self.tissue_population) < (self.start_population**2 / 2):
            return False

        # take care of cells
        if self.opti:
            self._tick_cell_programs_batch(absolute_time, relative_time)
        else:
            self._tick_cell_programs(absolute_time, relative_time)
            
        self.tissue.tick(absolute_time, relative_time)

        self.monitor(absolute_time + relative_time)  # because after computation
        self.snapshots.append(dict(
            tissue=self.tissue.export(),
            tissue_ids=self.tissue_population.copy(),
        ))
        assert set(self.tissue.export().keys()) == set(self.tissue_population.keys())
        return True

    def _tick_cell_programs(self, absolute_time, relative_time):
        # take care of cells
        for C_id in list(self.tissue_population.values()):
            C = self.population[C_id]
            neighbours = self.get_neighbours(C)
            action = C.tick(absolute_time, relative_time, neighbours)
            self.run_action(action, C, absolute_time)

    def _tick_cell_programs_batch(self, absolute_time, relative_time):
        # take care of cells
        cells = [self.population[C_id] for C_id in list(self.tissue_population.values())]
        for C in cells:
            neighbours = self.get_neighbours(C)
            C.set_neighbourhood(neighbours)

        # this batch function must be provided by the cell class
        self.cell_cls.batch_tick(cells, absolute_time, relative_time)

        for C in cells:
            action = C.get_action()
            self.run_action(action, C, absolute_time)

    def _sanity_check(self):
        if len(self.tissue_population) > self.max_pop_size:
            # raise RuntimeError("Model stops when population increases above {}".format(self.max_pop_size))
            self.info("Population explosion")
            return False
        return True
    
    def monitor(self, absolute_time):
        # monitor
        stats = dict(
            progenitor_pop_size=len(self.tissue_population),
            whole_pop_size=len(self.tissue_population) + len(self.post_mitotic),
            time=absolute_time,
            **self.population_size_by_type(),
        )

        self.add_stat_time(absolute_time, stats)

    #################
    ###  ACTIONS  ###
    #################

    def run_action(self, action, cell, T):
        self.mapping[action](cell, T)

    def divide(self, cell, T):
        """
        Warning : the id of one daughter is the same as for the mother
        We will have to modify the tissue code to overcome that
        """
        self.debug("Duplicating cell " + str(cell.index) + " located in " + str(cell.tissue_id))
        del self.tissue_population[cell.tissue_id]
        new_tissue_id1, new_tissue_id2 = self.tissue.divide_cell(cell.tissue_id)

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
        self.tissue.remove_cell(cell.tissue_id)
        del self.tissue_population[cell.tissue_id]

    def register_cell(self, cell):
        self.population[cell.index] = cell
        self.tissue_population[cell.tissue_id] = cell.index

    #################
    ###  HELPERS  ###
    #################

    def get_neighbours(self, cell):
        ngbs = self.tissue.get_neighbours(cell.tissue_id)
        return [self.population[self.tissue_population[i]] for i in ngbs]
    
    def get_neighbours_from_tissue_id(self, tissue_id, exclude=[]):
        ngbs = self.tissue.get_neighbours(tissue_id)
        ngbs = list(set(ngbs) - set(exclude))
        return [self.population[self.tissue_population[i]] for i in ngbs]
    
    def get_ngbs_types(self, tissue_id):
        ls_types = []
        ngbs = self.tissue.get_neighbours(tissue_id)
        for ngb in ngbs:
            if ngb in self.tissue_population:
                ls_types.append(self.population[self.tissue_population[ngb]].type())
       
        return ls_types

    def new_cell_id(self):
        self.id_count += 1
        return self.id_count
    
    def cell_from_tissue_id(self, tissue_id):
        return self.population[self.tissue_population[tissue_id]]

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
        
    @lru_cache
    def build_cell_history(self):
        df = pd.DataFrame()
        for c in self.population.values():
            row = dict(
                index=c.index,
                child1=c.children[0] if 0 < len(c.children) else None,
                child2=c.children[1] if 1 < len(c.children) else None,
                appear_time=c.appear_time,
                division_time=c.division_time,
                type=c.type().name,
                Tc_h=c.Tc_h(),
            )
            df = df.append(row, ignore_index=True)
        return df

    def add_stat_time(self, abs_time, stats: dict):
        self.stats = self.stats.append(pd.Series(stats, name=abs_time))

    def population_size_by_type(self, in_list=False):
        d = dict()
        count = Counter([self.population[index].type() for index
            in self.tissue_population.values()])
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
    def __init__(self, T, start=False, brain=None, index=None, tissue_id=None,
            parent=None, submodel=None, **kwargs):
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
                parent_type=self.type(), **kwargs)
