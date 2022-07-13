from jf.autocompute.jf import O
import pandas as pd
import numpy as np


REF = O(
    stats=pd.read_csv("reference/ref_tristate2.csv"),  # ref is a mean
)
    
    
callback_init = dict(
    init=lambda: np.random.beta(1.5, 3) * 3,
    b=lambda: np.random.beta(1.5, 3) * 5,
    expr=lambda: 1,
    deg=lambda: 0.1,
    noise=lambda: np.random.beta(1.5, 3) * 1,
    asym=lambda: 5,
)  


class BaseParam(O):
    pop_size = 50
    batch_size = 50
    n_gen = 50
    current_gen = 0
    end_time = 86
    ref = REF
    min_pop = 30
    max_pop = 50
    ts = 0.5
    start_time = 50
    model = "grn5"
    max_cell_pop_size = 5e2
    new_fitness = 0.05
    cb_init = callback_init
    n_genes = 5
    n_intergenes = 1
    size = 5
 
    
