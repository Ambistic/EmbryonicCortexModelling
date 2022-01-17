import numpy as np
import matplotlib.pyplot as plt

# function to consistently set a coeff to a proportion
# the basic rationale is that 1.1 * 0.1 increase only 0.1
# while 1.1 * 0.9 * 0.99 which is not comparable
# so this formula comes from passing a proportion p to x / (x + y)
# and setting a coeff to x, therefore giving p' = cx / (cx + y)
variate_prop = lambda p, c: c / (c - 1 + 1 / p)


def highest_lower(ls, x):
    return np.max(np.where(np.array(ls) <= x)[0])


nop = lambda *a, **k: None


def plot_function(timesteps, *funcs, figure=False):
    for f in funcs:
        if figure:
            plt.figure()
        plt.plot(timesteps, np.array([f(x) for x in timesteps]))


class Profiler:
    def __enter__(self):
        import cProfile

        self.pr = cProfile.Profile()
        self.pr.enable()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        import pstats, io
        self.pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(self.pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
