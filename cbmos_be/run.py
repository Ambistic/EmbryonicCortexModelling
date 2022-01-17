# coding: utf-8
# imports
import numpy as np
import pickle
import random
import pandas as pd
from model import Brain
from submodels import factories
import argparse
import os
from pathlib import Path
from traceback import print_exc


def go(args):
    smooth = args.smooth
    start_val = args.startval
    bias_ratio = [args.bias_ratio_1, args.bias_ratio_2, args.bias_ratio_3,
            args.bias_ratio_4, args.bias_ratio_5]
    tc_coeff = [args.tc_coeff_1, args.tc_coeff_2, args.tc_coeff_3,
            args.tc_coeff_4, args.tc_coeff_5]
    tc_coeff_RG = [args.tc_coeff_RG_1, args.tc_coeff_RG_2, args.tc_coeff_RG_3, args.tc_coeff_RG_4, args.tc_coeff_RG_5]
    tc_coeff_IP = [args.tc_coeff_IP_1, args.tc_coeff_IP_2, args.tc_coeff_IP_3, args.tc_coeff_IP_4, args.tc_coeff_IP_5]
    diff_values = [args.diff_values_1, args.diff_values_2, args.diff_values_3, args.diff_values_4, args.diff_values_5]
    diff_values_IP = [args.diff_values_IP_1, args.diff_values_IP_2, args.diff_values_IP_3,
                      args.diff_values_IP_4, args.diff_values_IP_5]
    diff_values_RG = [args.diff_values_RG_1, args.diff_values_RG_2, args.diff_values_RG_3,
                      args.diff_values_RG_4, args.diff_values_RG_5]
    diff_values_RG_IP = [args.diff_values_RG_IP_1, args.diff_values_RG_IP_2, args.diff_values_RG_IP_3,
                         args.diff_values_RG_IP_4, args.diff_values_RG_IP_5]
    diff_values_RG_GP = [args.diff_values_RG_GP_1, args.diff_values_RG_GP_2, args.diff_values_RG_GP_3,
                         args.diff_values_RG_GP_4, args.diff_values_RG_GP_5]
    
    size = args.size
    GP_as_IP = args.gp_as_ip
    DESC = f"bias {bias_ratio} size {size} startval {start_val} smooth {smooth}" \
            f"gp_as_ip : {GP_as_IP}"
    kwargs = dict(smooth=smooth, start_val=start_val, bias_ratio=bias_ratio, tc_coeff=tc_coeff,
                  tc_coeff_RG=tc_coeff_RG, tc_coeff_IP=tc_coeff_IP, diff_values=diff_values,
                  diff_values_IP=diff_values_IP, diff_values_RG=diff_values_RG,
                  diff_values_RG_IP=diff_values_RG_IP, diff_values_RG_GP=diff_values_RG_GP,
                 )
    ccls = factories[args.model](**kwargs)
    print(kwargs)

    bb = Brain(time_step=0.5, verbose=False, start_population=size,
            cell_cls=ccls.generate, end_time=args.end)
    bb.run()
    save_network_snapshots(bb.snapshots, args.name)
    save_stats(bb.stats, args.name)
    save_history(bb.population, args.name)
    save_txt(DESC, args.name)
    return bb


def build_cell_history(pop):
    df = pd.DataFrame()
    for c in pop.values():
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

def save_network_snapshots(snapshots, name):
    with open(ROOT / f"output/results/snapshots_{name}.pck", "wb") as f:
        pickle.dump(snapshots, f)

def save_history(pop, name):
    history = build_cell_history(pop)
    history.to_csv(str(ROOT / f"output/results/history_{name}.csv"))
        
def save_stats(stats, name):
    stats.to_csv(str(ROOT / f"output/results/stats_{name}.csv"))
    
def save_txt(txt, name):
    fn = str(ROOT / f"output/results/txt_{name}.txt")
    with open(fn, "w") as f:
        f.write(txt)
        
def register_float_args(parser, inputs):
    for name, default in inputs:
        parser.add_argument("--" + name, default=default, type=float)


if __name__ == "__main__":
    ROOT = Path(os.path.realpath(__file__)).parent.parent
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--smooth", default=0.0, type=float)
    parser.add_argument("-n", "--name", default=None, type=str)
    parser.add_argument("-v", "--startval", default=0.35, type=float)
    parser.add_argument("-p", "--sample", default=1, type=int)
    parser.add_argument("-s", "--size", default=6, type=int)
    parser.add_argument("-e", "--end", default=80, type=int)
    parser.add_argument("-g", "--gp-as-ip", action="store_true")
    parser.add_argument("-m", "--model", default="basic", type=str)
    parser.add_argument("-c", "--check", action="store_true")
    
    float_list = [
        ("bias_ratio_1", 0.2),
        ("bias_ratio_2", 0.15),
        ("bias_ratio_3", -0.1),
        ("bias_ratio_4", -0.1),
        ("bias_ratio_5", -0.1),
        
        ("tc_coeff_1", 1.),
        ("tc_coeff_2", 1.),
        ("tc_coeff_3", 1.),
        ("tc_coeff_4", 1.),
        ("tc_coeff_5", 1.),
        
        ("tc_coeff_RG_1", 1.),
        ("tc_coeff_RG_2", 1.),
        ("tc_coeff_RG_3", 1.),
        ("tc_coeff_RG_4", 1.),
        ("tc_coeff_RG_5", 1.),
        
        ("tc_coeff_IP_1", 1.),
        ("tc_coeff_IP_2", 1.),
        ("tc_coeff_IP_3", 1.),
        ("tc_coeff_IP_4", 1.),
        ("tc_coeff_IP_5", 1.),
        
        ("diff_values_1", 0.73),
        ("diff_values_2", 0.63),
        ("diff_values_3", 0.47),
        ("diff_values_4", 0.45),
        ("diff_values_5", 0.45),
        
        ("diff_values_IP_1", 0.23),
        ("diff_values_IP_2", 0.23),
        ("diff_values_IP_3", 0.23),
        ("diff_values_IP_4", 0.23),
        ("diff_values_IP_5", 0.23),
        
        ("diff_values_RG_1", 0.63),
        ("diff_values_RG_2", 0.53),
        ("diff_values_RG_3", 0.43),
        ("diff_values_RG_4", 0.38),
        ("diff_values_RG_5", 0.33),
        
        ("diff_values_RG_IP_1", 0.63),
        ("diff_values_RG_IP_2", 0.53),
        ("diff_values_RG_IP_3", 0.43),
        ("diff_values_RG_IP_4", 0.38),
        ("diff_values_RG_IP_5", 0.33),
        
        ("diff_values_RG_GP_1", 1.),
        ("diff_values_RG_GP_2", 1.),
        ("diff_values_RG_GP_3", 1.),
        ("diff_values_RG_GP_4", 0.8),
        ("diff_values_RG_GP_5", 0.6),  
    ]
    
    register_float_args(parser, float_list)


    args = parser.parse_args()

    if args.name is None:
        args.name = str(random.randint(0, 1e6))    
    
    seed = 0
    while True:
        print("Doing", seed + int(args.sample * 1e3))
        random.seed(seed + int(args.sample * 1e3))
        np.random.seed(seed + int(args.sample * 1e3))
        try:
            bb = go(args)
        except Exception as e:
            print("Exception found", e)
            print("Traceback below")
            print_exc()
            seed += 1
            if seed > 5:  # after 5, it means that we seriously should consider investigating the bugs
                raise
        else:
            break

