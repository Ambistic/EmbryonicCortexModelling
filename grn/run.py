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
import re


def get_params():
    return dict(
        tc_coeff_RG=[1., 1., 1., 1., 1.],
        tc_coeff_IP=[1., 1., 1., 1., 1.],
        diff_values_RG_GP=[1., 1., 1., 0.8, 0.6], # prob GP vs IP
        diff_values_RG_IP=[.7, .6, .4, 0.45, 0.4], # prob IP vs renew
        diff_values_IP=[0.23, 0.23, 0.23, 0.23, 0.23],
        bias_ratio=[0.7, 0.62, 0.33, 0.5, 0.5],  # the higher is the bias, the more there are RG
        smooth=0.0,
        start_val=0.4,
        id_mutant=-1,
        GP_as_IP=False,
    )

def is_float(string):
    try:
        float(string)
    except ValueError:  # String is not a number
        return False
    else:
        return '.' in string

def parse_getitem(name):
    n, i = re.findall("^([^[]*)\[(\d+)\]", name)[0]
    return n, int(i)

def parse_value(value):
    print("PARSING", value)
    if value == "True":
        return True
    
    elif value == "False":
        return False
    
    elif value.isdigit():
        return int(value)
    
    elif is_float(value):
        return float(value)
    
    return value

def parse_one_param(ref_params, p):
    if not len(p) == 2:
        raise ValueError(f"Must provide a name and a value, {p}")
        
    name, value = p
    
    value = parse_value(value)
    print("PARSED", value, type(value))
    
    if "[" in name:
        pname, idx = parse_getitem(name)
        ref_params[pname][idx] = value
        
    else:
        ref_params[name] = value

def parse_params(ref_params, params):
    for p in params:
        parse_one_param(ref_params, p)

def go(args):
    # TODO new param eval technique
    params = get_params()
    parse_params(params, args.params)  # side effect
    
    DESC = f"List of params {params}, size {args.size}, end {args.end}"

    ccls = factories[args.model](**params)

    bb = Brain(time_step=0.5, verbose=False, start_population=args.size,
            cell_cls=ccls.generate, end_time=args.end, start_time=args.start)
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
    with open(ROOT / f"output/results/{name}.snapshots.pck", "wb") as f:
        pickle.dump(snapshots, f)

def save_history(pop, name):
    history = build_cell_history(pop)
    history.to_csv(str(ROOT / f"output/results/{name}.history.csv"))

    
def save_stats(stats, name):
    stats.to_csv(str(ROOT / f"output/results/{name}.stats.csv"))

    
def save_txt(txt, name):
    fn = str(ROOT / f"output/results/{name}.txt.txt")
    with open(fn, "w") as f:
        f.write(txt)


if __name__ == "__main__":
    ROOT = Path(os.path.realpath(__file__)).parent.parent
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--name", default=None, type=str)
    parser.add_argument("-i", "--sample", default=1, type=int)
    parser.add_argument("-s", "--size", default=6, type=int)
    parser.add_argument("-e", "--end", default=80, type=int)
    parser.add_argument("-t", "--start", default=49, type=int)
    parser.add_argument("-m", "--model", default="basic", type=str)
    parser.add_argument("-c", "--check", action="store_true")
    parser.add_argument("-p", "--params", nargs="*", action='append', default=[])

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
            if seed > 0:  # after 5, it means that we seriously should consider investigating the bugs
                raise
        else:
            break

