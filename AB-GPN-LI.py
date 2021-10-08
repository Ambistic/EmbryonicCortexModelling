# coding: utf-8
# imports
import numpy as np
import seaborn as sns
import tqdm
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
from numpy.random import choice
from scipy import stats
from collections import Counter
from model import AbstractCell, Action, Brain, Submodels
from modelrgpn import BrainLI
from experiment import Experiment
from utils import nop, highest_lower, Profiler, plot_function
from submodels.cellbasic1 import CellBasic
from submodels.bistate1 import BiStateModelFactory
from submodels.bistate_LI import BiStateLIModelFactory
from submodels.tristate_LI import TriStateLIModelFactory
from submodels import factories
from biodata import *
from tree import tree_from_cell
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("-o", "--smooth", default=0.0, type=float)
parser.add_argument("-n", "--name", default=None, type=str)
parser.add_argument("-v", "--startval", default=0.35, type=float)
parser.add_argument("-p", "--sample", default=1, type=int)
parser.add_argument("-s", "--size", default=6, type=int)
parser.add_argument("-e", "--end", default=80, type=int)
parser.add_argument("-g", "--gp-as-ip", action="store_true")

parser.add_argument("--bias_1", default=0.2, type=float)
parser.add_argument("--bias_2", default=0.2, type=float)
parser.add_argument("--bias_3", default=-0.1, type=float)
parser.add_argument("--bias_4", default=-0.1, type=float)
parser.add_argument("--bias_5", default=-0.1, type=float)

parser.add_argument("--check", action="store_true")

args = parser.parse_args()

if args.name is None:
    args.name = str(random.randint(0, 1e6))

# In[3]:


# Definition of the var of the model
# We start with 100 cells for homogeneity
START_POPULATION_SQRT = 10
START_TIME = 49
END_TIME = 94
# We arbitrarily set variance of Tc at 12.5^2
# 95 % seems to be into 50, so sigma = 50 / 4 = 12.5
SIG_TC = 12.5


# In[4]:


smooth = args.smooth
start_val = args.startval
bias_ratio = [args.bias_1, args.bias_2, args.bias_3, args.bias_4, args.bias_5]
size = args.size
NAME = args.name
GP_AS_IP = args.gp_as_ip
DESC = f"bias {bias_ratio} size {size} startval {start_val} smooth {smooth}" \
        f"gp_as_ip : {GP_AS_IP}"


# In[5]:


def go(seed=0):
    random.seed(0+seed)
    np.random.seed(0+seed)
    ccls = TriStateLIModelFactory(tc_coeff_RG=[1., 1., 1., 1., 1.],
            smooth=smooth, start_val=start_val,
            bias_ratio=bias_ratio, GP_as_IP=GP_AS_IP)

    bb = BrainLI(time_step=0.5, verbose=True, start_population=size,
            cell_cls=ccls.generate, check=args.check,
            end_time=args.end)
    print(bb.gpn)
    bb.run()
    return bb


# In[ ]:


seed = 0
while True:
    print("Doing", seed)
    try:
        bb = go(seed + int(args.sample * 1e3))
    except:
        seed += 1
        if seed > 15:
            raise
    else:
        break


# In[ ]:


def plot_size(stats, name=None):
    ref = stats.whole_pop_size.iloc[0]
    plot_number_cells()
    p1 = plt.plot(stats.time, stats.progenitor_pop_size / ref, label="Progenitor population")
    # p2 = plt.plot(stats.time, stats.whole_pop_size / ref, label="Whole Population")
    plt.legend()
    if name:
        plt.savefig(f"output/results/prog_size_{name}.png")
    
def plot_ratio(stats, name=None):
    stats = stats.fillna(0)
    non_IP = stats.size_type_RG + stats.size_type_GP if "size_type_GP" in stats.columns else stats.size_type_RG
    
    plt.plot(stats.time, stats.size_type_IP / (non_IP + stats.size_type_IP),
             label="IP ratio")
    plt.plot(ratio_eomes.index, ratio_eomes.val / 100, label="Reference IP ratio")
    plt.legend()
    if name:
        plt.savefig(f"output/results/IP_ratio_{name}.png")
        
def plot_size_ratio(stats, name=None):
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plot_size(stats)
    plt.subplot(2, 1, 2)
    plot_ratio(stats)
    if name:
        plt.savefig(f"output/results/prog_size_ratio_{name}.png")
        
def save_stats(stats, name):
    stats.to_csv(f"output/results/stats_{name}.csv")
    
def save_txt(txt, name):
    fn = f"output/results/txt_{name}.txt"
    with open(fn, "w") as f:
        f.write(txt)


# In[ ]:


plot_size_ratio(bb.stats, NAME)


# In[ ]:


save_stats(bb.stats, NAME)
save_txt(DESC, NAME)

