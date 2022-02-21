class Objective:
    id_func = 0
    id_max_trial = 1
    id_score_threshold = 2

    def __init__(self, objectives):
        self.objectives = objectives
        self.total_trial = 0
        self.current_nb_trial = 0
        self.current = 0

    def reset(self):
        self.current = 0
        self.total_trial = 0
        self.current_nb_trial = 0

    def get_objective_func(self):
        return self.objectives[self.current][self.id_func]

    def new_trial(self):
        self.current_nb_trial += 1
        self.total_trial += 1
        if self.current_nb_trial >= self.objectives[self.current][self.id_max_trial]:
            self._next()

    def best_current(self, score):
        if score >= self.objectives[self.current][self.id_score_threshold]:
            self._next()

    def _next(self):
        self.current += 1
        if self.current >= len(self.objectives):
            self.current = len(self.objectives) - 1
        else:
            print("Going to next objective !")
            self.current_nb_trial = 0
            