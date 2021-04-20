from ete3 import Tree, NodeStyle
from collections import Counter
import pandas as pd
import numpy as np

style_RG = NodeStyle()
style_RG["fgcolor"] = "purple"
style_RG["size"] = 15

style_IP = NodeStyle()
style_IP["fgcolor"] = "green"
style_IP["size"] = 15

style_N = NodeStyle()
style_N["fgcolor"] = "red"
style_N["size"] = 15

style_GP = NodeStyle()
style_GP["fgcolor"] = "yellow"
style_GP["size"] = 15

style_default = NodeStyle()


style = dict(RG=style_RG, IP=style_IP, PostMitotic=style_N, GP=style_GP)


def recursive_add(tree, cell):
    n = tree.add_child(name=cell.index, dist=cell.Tc_h())
    n.set_style(style.get(cell.type().name, style_default))
    for child in cell.get_children():
        recursive_add(n, child)


def tree_from_cell(cell):
    tree = Tree()
    
    recursive_add(tree, cell)
    return tree



def count_total_progeny(cell, brain, only_leaves=True):
    if isinstance(cell, int):
        cell = brain.population[cell]
        
    tree = tree_from_cell(cell)
    if only_leaves:
        nodes = tree.get_leaves()
    else:
        nodes = list(tree.traverse())
        nodes = [n for n in nodes if isinstance(n.name, int)]
        
    res = Counter(list(map(lambda x: brain.population[x.name].type(), nodes)))
    return res


def df_cells(pop):
    times = list(map(lambda x: x.appear_time, pop.values()))
    types = list(map(lambda x: x.type().name, pop.values()))
    index = list(map(lambda x: x.index, pop.values()))
    return pd.DataFrame({"cell": index, "appear_time": times, "type": types})


def progeny_along_time(brain, _type="RG", only_leaves=True):
    df = df_cells(brain.population)
    
    ret_df = pd.DataFrame()
    
    times = sorted(df["appear_time"].unique())
    for T in times:
        count_T = Counter()
        df_T = df[(df["appear_time"] == T) & (df["type"] == _type)]
        N = len(df_T)
        if not N:
            continue
            
        for i, cell in df_T.iterrows():
            progeny = count_total_progeny(cell["cell"], brain, only_leaves=only_leaves)
            count_T.update(progeny)
            
        new_row = dict(count_T)
        new_row["time"] = T
        ret_df = ret_df.append(new_row, ignore_index=True)
        
    ret_df = ret_df.fillna(0.0)
        
    return ret_df