import argparse
from pathlib import Path as P
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from vizualisation.utils import gather_groups, gather_metagroups, read_dir, \
    listtuple_to_dict, get_param_name, get_name_for_file
from vizualisation.plot import renderers

def render(args):
    params = read_dir(args.dir)
    groups = gather_groups(params)
    metagroups = gather_metagroups(groups)

    print("Building metagroups :", *list(metagroups.keys()))
    
    # save global csv file
    for key, meta in tqdm(metagroups.items()):
        for renderer in renderers:
            title = " ".join([get_param_name(key), renderer.name])
            renderer(meta, args.dir, title=title)
            plt.savefig(P(args.outdir) / (get_name_for_file(key) + "_" + renderer.name + ".png"))
            plt.close()


if __name__ == "__main__":
    print(__file__, P(__file__).parent, P(".").parent)
    ROOT = P(os.path.realpath(__file__)).parent.parent
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--dir", type=str, default="output/results")
    parser.add_argument("-o", "--outdir", type=str, default="output/results/export")
    
    # group by default
    
    args = parser.parse_args()
    P(args.outdir).mkdir(parents=True, exist_ok=True)
    print("Let's go !!")
    render(args)
    print("Finished !")
    