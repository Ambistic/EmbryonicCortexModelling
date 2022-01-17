from model import *


class BrainLI(Brain):
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
        callback_gpn=None
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
        self.gpn = RelaxableGPN(callback=callback_gpn)
        self.init_mapping()
        self.initiate_population()
        self.init_stats()
        self.snapshots = list()
        
    def initiate_population(self):
        self.gpn.init_tissue(self.start_population)
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
        
    def divide(self, cell, T):
        self._divide(cell, T)
            
    def remove_cell(self, cell, T):
        self._remove_cell(cell, T)
            
    def _remove_cell(self, cell, T):
        self.debug("Removing " + str(cell.index) + " " + str(cell.gpn_id))
        self.gpn.destroy(cell.gpn_id, relax=True)
        if self.check:
            self.gpn.check_all()

        del self.gpn_population[cell.gpn_id]
        
    def _divide(self, cell, T):
        """
        Warning : the id of one daughter is the same as for the mother
        We will have to modify the gpn code to overcome that
        """
        self.debug("Duplicating " + str(cell.index) + " " + str(cell.gpn_id))
        new_gpn_id = self.gpn.duplicate(cell.gpn_id, relax=True)
        time_ = cell.appear_time + cell.eff_Tc
        # time_ = T
        new_cell_1 = cell.generate_daughter_cell(time_, index=self.new_cell_id(),
                gpn_id=new_gpn_id)
        new_cell_2 = cell.generate_daughter_cell(time_, index=self.new_cell_id(),
                gpn_id=cell.gpn_id)
        if self.check:
            self.gpn.check_all()
            
        self.debug(f"Duplicated with ids : {cell.gpn_id} and {new_gpn_id}")

        self.register_cell(new_cell_1)
        self.register_cell(new_cell_2)
