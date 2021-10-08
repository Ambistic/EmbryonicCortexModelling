import os
import pandas as pd
from pathlib import Path as P

def get_df():
    pass

def generate_single_curve(f):
    df = pd.read_csv(f)
    print(df.head())
    print(df.columns)

def pick_modalities(var):
    root = P("output/results")
    mod = []
    for f in filter(lambda x: ".csv" in x,
            os.listdir(root)):
        if var in f:
            mod.append(generate_single_curve(root / f))

    return mod

def make_viz(mod):
    pass

def export_viz(viz, path):
    pass

def build_viz(root):
    variates = [
        "gpasip",
        "smooth",
        "startval",
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
    ]

    for var in variates:
        mod = pick_modalities(var)
        viz = make_viz(mod)
        export_viz(viz, root / (var + ".png"))
        break


if __name__ == "__main__":
    build_viz(P("output"))
