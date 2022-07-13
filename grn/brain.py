import numpy as np
import pandas as pd

from lib.population import Base, CellPopulation
from functools import lru_cache
from collections import defaultdict

from cbmos import CBModelv3
import cbmos.force_functions as ff
import cbmos.solvers.euler_forward as ef

START_POPULATION_SQRT = 10
START_TIME = 49
END_TIME = 94
SIG_TC = 12.5


def get_tissue():
    dt = 0.01
    mu = 5.70
    s = 1.0
    a = 5.0
    rA = 1.5
    return CBModelv3(ff.Gls(), ef.solve_ivp, dimension=2,
                     force_args={"mu": mu, "s": s, "rA": rA, "a": a},
                     solver_args={'dt': dt}, )


class Submodels:
    """
    This class implements all parameters that are required, also under the callback form
    """

    def __init__(self, Tc_func, diff_func, repr_=None, start_diff=None):
        self.Tc = Tc_func
        self.diff = diff_func
        self.repr_ = repr_
        self.start_diff = start_diff


# Definition of the class
class BrainModel(Base):
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
            cell_cls=None,
            verbose=False,
            silent=False,
            opti=False,
            max_pop_size=1e4,
            record_tissue=False,
            record_population=False,
            run_tissue=True,
            monitor_callbacks=None,
            tag_func=None
    ):
        # debug args
        if monitor_callbacks is None:
            monitor_callbacks = dict()
        self.monitor_callbacks = monitor_callbacks
        self.run_tissue = run_tissue
        self.record_tissue = record_tissue
        self.record_population = record_population
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
        self.range = np.arange(self.start_time, self.end_time, self.time_step)
        self.lenrange = len(self.range)
        self.current_step = 0
        self.max_pop_size = max_pop_size

        # init
        self.tissue = get_tissue()
        self.tissue.init_square(self.start_population)
        self.population: CellPopulation = CellPopulation(self, self.tissue.get_ids(),
                                                         self.cell_cls, self.start_time)
        if tag_func is not None:
            tag_func(self.population)

        self.stats = pd.DataFrame({})
        self.snapshots = defaultdict(dict)
        self.monitor(self.start_time)
        if self.record_population:
            self.create_snapshot_population(self.start_time)

        if self.record_tissue:
            self.create_snapshot_tissue(self.start_time)

    @staticmethod
    def build_cell_cls(cell_cls):
        from submodels import factories
        if callable(cell_cls):
            return cell_cls

        elif isinstance(cell_cls, tuple):
            name, args, kwargs = cell_cls
            factory = factories[name](*args, **kwargs)
            return factory

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
        size = len(self.population.tissue_population)
        self.info(f"Ticking abs : {absolute_time}, step : {relative_time}, size : {size}")
        if not self._sanity_check():
            return False

        # take care of cells
        if self.opti:
            self.population.tick_cell_programs_batch(absolute_time, relative_time)
        else:
            self.population.tick_cell_programs(absolute_time, relative_time)

        if self.run_tissue:
            self.tissue.tick(absolute_time, relative_time)

        self.monitor(absolute_time + relative_time)  # because after computation
        if self.record_population:
            self.create_snapshot_population(absolute_time + relative_time)

        if self.record_tissue:
            self.create_snapshot_tissue(absolute_time + relative_time)

        return True

    def _sanity_check(self):
        if len(self.population.tissue_population) > self.max_pop_size:
            # raise RuntimeError("Model stops when population increases above {}".format(self.max_pop_size))
            self.info("Population explosion")
            return False

        elif len(self.population.tissue_population) < (self.start_population ** 2 / 2):
            self.info("Population exhausted")
            return False

        return True

    def monitor_old(self, absolute_time):
        # monitor
        stats = dict(
            progenitor_pop_size=len(self.population.tissue_population),
            whole_pop_size=len(self.population.tissue_population) + len(self.population.post_mitotic),
            time=absolute_time,
        )

        self.add_stat_time(absolute_time, stats)

    def monitor(self, absolute_time):
        stats = {name: callback(self.population) for name, callback in self.monitor_callbacks.items()}
        stats["time"] = absolute_time
        self.add_stat_time(absolute_time, stats)

    #################
    ###  HELPERS  ###
    #################

    def get_neighbours(self, cell):
        ngbs = self.tissue.get_neighbours(cell.tissue_id)
        return [self.population.cell_from_tissue_id(i) for i in ngbs]

    def get_neighbours_from_tissue_id(self, tissue_id, exclude=None):
        if exclude is None:
            exclude = []
        ngbs = self.tissue.get_neighbours(tissue_id)
        ngbs = list(set(ngbs) - set(exclude))
        return [self.population.cell_from_tissue_id(i) for i in ngbs]

    def get_ngbs_types(self, tissue_id):
        ls_types = []
        ngbs = self.tissue.get_neighbours(tissue_id)
        for ngb in ngbs:
            if ngb in self.population.tissue_population:
                ls_types.append(self.population.cell_from_tissue_id(ngb).type())

        return ls_types

    ##################
    ### STATISTICS ###
    ##################

    @lru_cache
    def build_cell_history(self):
        df = pd.DataFrame()
        for c in self.population.base_population.values():
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

    def create_snapshot_population(self, time):
        dict_snap_population = dict()
        for C_id in list(self.population.tissue_population.values()):
            C = self.population.base_population[C_id]
            dict_snap_population[C_id] = C.freeze()

        self.snapshots[time]["population"] = dict_snap_population

    def create_snapshot_tissue(self, time):
        self.snapshots[time]["tissue"] = self.tissue.export()
        self.snapshots[time]["tissue_ids"] = self.population.tissue_population.copy()
