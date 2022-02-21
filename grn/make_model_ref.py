import argparse
import pandas as pd
from collections import defaultdict
from lib.score import fate_corr
from lib.preprocess import get_fmetric_pairs
from lib.utils import mean_dict
from jf.autocompute.jf import O


class TmpCell:
    def __init__(self, appear_time, cell_type: str, children: list = []):
        if len(children) == 2:
            self._type = O(name="Progenitor")
        elif cell_type == "PostMitotic":
            self._type = O(name="PostMitotic")
        else:
            self._type = O(name="Cell")
        
        self.children = children
        self.appear_time = appear_time
        
    def type(self):
        return self._type
    

def build_population(df):
    population = dict()
    for i, row in df.iterrows():
        children = list(map(int, filter(lambda x: not pd.isna(x), [row["child1"], row["child2"]])))
        population[int(row["index"])] = TmpCell(row["appear_time"], row["type"], children)
        
    return population


def multiple_fate_corr(population):
    return dict(
        fate_corr_50_60=fate_corr(get_fmetric_pairs(population, 50, 60)),
        fate_corr_60_70=fate_corr(get_fmetric_pairs(population, 60, 70)),
        fate_corr_70_80=fate_corr(get_fmetric_pairs(population, 70, 80)),
        fate_corr_80_80=fate_corr(get_fmetric_pairs(population, 80, 90)),
        fate_corr_50_70=fate_corr(get_fmetric_pairs(population, 50, 70)),
        fate_corr_50_80=fate_corr(get_fmetric_pairs(population, 50, 80)),
        fate_corr_50_90=fate_corr(get_fmetric_pairs(population, 50, 90)),
    )

def main_fmetric(args):
    """ Requires history"""
    ls_scores = []
    for f in args.files:
        df = pd.read_csv(f)
        population = build_population(df)
        scores = multiple_fate_corr(population)
        ls_scores.append(scores)
    
    final_scores = mean_dict(*ls_scores)
    pd.Series(final_scores).to_csv(args.output)


def main_size(args):
    def_dict = defaultdict(list)
    cols = {"size_type_Cycling", "size_type_PostMitotic",
           "whole_pop_size", "progenitor_pop_size"}
    for f in args.files:
        df = pd.read_csv(f).set_index("time")
        for col in df.columns:
            if not col in cols:
                continue
            def_dict[col].append(df[col])

    # mean
    mean_series = list()
    for k, v in def_dict.items():
        df = pd.concat(v).fillna(0)
        df_group = df.groupby(level=0).mean()

        mean_series.append(df_group)

    final_df = pd.concat(mean_series, axis=1).reset_index().copy()
    final_df["neuron_pop_size"] = final_df["whole_pop_size"] - final_df["progenitor_pop_size"]
    final_df.to_csv(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=['size', 'fmetric'])
    parser.add_argument("-f", "--files", nargs="*")
    parser.add_argument("-o", "--output")

    args = parser.parse_args()
    if args.action == "size":
        main_size(args)
        
    elif args.action == "fmetric":
        main_fmetric(args)
