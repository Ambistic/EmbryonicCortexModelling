import numpy as np
import pandas as pd
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


def shrink_and_align_stats(stats, ref, max_step=None, min_step=None):
    min_data_step = max(min(stats.time), min(ref.time))
    if min_step is None:
        min_step = min_data_step
    else:
        min_step = max(min_step, min_data_step)

    if max_step is None:
        max_step = max(ref.time)
    else:
        max_step = min(max_step, max(ref.time))

    new_stats = stats[(stats.time >= min_step) & (stats.time <= max_step)].set_index("time")
    new_ref = ref[(ref.time >= min_step) & (ref.time <= max_step)].set_index("time")

    return new_stats, new_ref


def align_time(df_data, df_ref):
    min_step = max(min(df_data.time), min(df_ref.time))
    max_step = max(df_ref.time)

    new_df_data = df_data[(df_data.time >= min_step) & (df_data.time <= max_step)]
    new_df_ref = df_ref[(df_ref.time >= min_step) & (df_ref.time <= max_step)]

    return new_df_data, new_df_ref


def normalize_time(df_data, df_ref, col_data, col_ref, col_norm_data=None, col_norm_ref=None):
    if col_norm_data is None:
        col_norm_data = col_data
        
    if col_norm_ref is None:
        col_norm_ref = col_ref
    # here get index of, not the value
    id_min_data = df_data[df_data.time == min(df_data.time)].index[0]
    id_min_ref = df_ref[df_ref.time == min(df_ref.time)].index[0]
    
    df_data, df_ref = df_data.copy(), df_ref.copy()
    df_data.loc[:, col_data] = df_data[col_data] / df_data[col_norm_data].get(id_min_data)
    df_ref.loc[:, col_ref] = df_ref[col_ref] / df_ref[col_norm_ref].get(id_min_ref)
    
    return df_data, df_ref


def shrink_time(df_data, df_ref, min_time=None, max_time=None):
    min_step = min_time if min_time is not None else min(df_ref.time)
    max_step = max_time if max_time is not None else max(df_ref.time)
    new_df_data = df_data[(df_data.time >= min_step) & (df_data.time <= max_step)]
    new_df_ref = df_ref[(df_ref.time >= min_step) & (df_ref.time <= max_step)]

    return new_df_data, new_df_ref


def as_time_lists(df_data, df_ref, col_data, col_ref):
    df_ref = df_ref.set_index("time")
    df_data = df_data.set_index("time")
    x = [df_ref[col_ref].get(t, 0) for t in df_ref.index]
    y = [df_data[col_data].get(t, 0) for t in df_ref.index]
    return x, y


def mean_dict(*dicts):
    return dict(pd.DataFrame(list(dicts)).mean())
