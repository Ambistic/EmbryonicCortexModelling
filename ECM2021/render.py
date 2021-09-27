import argparse
from pathlib import Path as P
import pandas as pd
import os

from viz import plot_size_ratio

def render(args):
    metrics = dict()  # name : dict(index=row)
    names = []
    
    # select group of files
    if args.name:
        names.append(args.name)
        
    if args.dir:
        # catch using stats_ prefix
        pass
        
    if len(names) == 0:
        print("No result to process")
    
    # select rendering functions
    pass  # later
    
    # execute
    for name in names:
        stats = get_stats(name)
        export_curves(stats, name)
    
    # mean + sd groups
    # process_dist(metrics)
    
    # save global csv file
    pass

def process_dist(metrics):
    for name, dist in metrics.items():
        # make dataframe
        df = pd.DataFrame(dist).transpose()
        pass  # unfinished

# Objects

def get_stats(name, directory="output/results"):
    fn = ROOT / directory / f"stats_{name}.csv"
    return pd.read_csv(fn)

def get_snapshots():
    pass

def get_history():
    pass

# Exports

def export_tlv_metrics():
    "Export csv file"
    pass

def export_network_sample():
    "Export images"
    pass

def export_network_metrics():
    # correlation between a node type and its neighbours
    pass

def export_curves(stats, name):  # most important
    plot_size_ratio(stats, name, root=ROOT)

def export_progeny_plot():
    pass

def export_progeny_metrics():
    pass

def export_tree_metrics():
    pass

def export_tree_plot():
    pass


if __name__ == "__main__":
    print(__file__, P(__file__).parent, P(".").parent)
    ROOT = P(os.path.realpath(__file__)).parent.parent
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-a", "--all", action="store_true")
    parser.add_argument("-n", "--name", type=str)
    parser.add_argument("-d", "--dir", type=str)
    parser.add_argument("-o", "--outdir", type=str, default="output/results")
    # group by default
    
    args = parser.parse_args()
    
    render(args)
    