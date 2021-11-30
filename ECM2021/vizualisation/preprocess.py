from ete3 import Tree
from pathlib import Path as P
from collections import Counter
import pandas as pd
from scipy import stats
import numpy as np

from .defaults import STYLE
from debug import debugger

def recursive_add(history, tree, cell_id):
    if cell_id == -1:
        return
    cell = history.loc[cell_id]
    node = tree.add_child(name=cell_id, dist=cell["Tc_h"])
    node.set_style(STYLE.get(cell["type"]))
    for child_id in cell[["child1", "child2"]]:
        recursive_add(history, node, child_id)

def tree_from_cell(history, cell_id):
    tree = Tree()
    recursive_add(history, tree, cell_id)
    return tree

# PREPROCESS
def preprocess_history(param, root):
    file = "history_" + param["name"] + ".csv"
    df = pd.read_csv(P(root) / file, index_col=0)
    df["index"] = df["index"].astype(int)
    df["child1"] = df["child1"].fillna(-1)
    df["child2"] = df["child2"].fillna(-1)
    df["child1"] = df["child1"].astype(int)
    df["child2"] = df["child2"].astype(int)
    df.set_index("index")
    return df

def preprocess_stats(param, root):
    file = "stats_" + param["name"] + ".csv"
    df = pd.read_csv(P(root) / file)
    return df

def count_total_progeny(history, cell_id, tree=None, only_leaves=True):
    if tree is None:
        tree = tree_from_cell(history, cell_id)
    if only_leaves:
        nodes = tree.get_leaves()
    else:
        nodes = list(filter(lambda x: isinstance(x.name, (int, np.integer)), tree.traverse()))
        
    return Counter(list(map(lambda x: history.loc[x.name]["type"], nodes)))

def progeny_along_time(history, _type="RG", only_leaves=True, step=2., appear_time=49.):
    ret_df = pd.DataFrame()
    
    min_time, max_time = min(history["appear_time"]), max(history["appear_time"])
    dict_trees = dict()
    # build trees
    for i, row in history.iterrows():
        if row["appear_time"] == appear_time:
            tree = tree_from_cell(history, row["index"])
            dict_trees.update({x.name: x for x in tree.traverse()})
            
    if _type not in history["type"].values:
        _type = "Cycling"
    
    for T in np.arange(min_time, max_time, step):
        count_T = Counter()
        df_T = history[(history["appear_time"] >= T) &
                  (history["appear_time"] < (T + step)) & (history["type"] == _type)]
        N = len(df_T)
        if not N:
            continue
            
        for i, cell in df_T.iterrows():
            progeny = count_total_progeny(history, cell["index"], dict_trees.get(cell["index"]), only_leaves=only_leaves)
            count_T.update(progeny)
            
        new_row = {k: v / N for k, v in dict(count_T).items()}
        new_row["time"] = T
        ret_df = ret_df.append(new_row, ignore_index=True)
        
    ret_df = ret_df.fillna(0.0)
        
    return ret_df

def preprocess_nb_progenitor(param, root):
    stats = preprocess_stats(param, root)
    ref = stats.whole_pop_size.iloc[0]
    return dict(x=stats.time, y=stats.progenitor_pop_size / ref)

def preprocess_nb_cells(param, root):
    stats = preprocess_stats(param, root)
    ref = stats.whole_pop_size.iloc[0]
    return dict(x=stats.time, y=stats.whole_pop_size / ref)

def preprocess_ratio(param, root):
    stats = preprocess_stats(param, root)
    stats = stats.fillna(0)
    if "size_type_Cycling" in stats.columns:
        non_IP = stats.size_type_Cycling
        IP = 0
    else:
        non_IP = stats.size_type_RG + stats.size_type_GP if "size_type_GP" in stats.columns else stats.size_type_RG
        IP = stats.size_type_IP
    
    return dict(x=stats.time, y=IP / (non_IP + IP))

def preprocess_progeny(param, root):
    history = preprocess_history(param, root)
    return progeny_along_time(history)

def preprocess_progeny_all(param, root):
    history = preprocess_history(param, root)
    return progeny_along_time(history, only_leaves=False)

def mean_sd(ls):
    min_len = min([len(sample["x"]) for sample in ls])
    x = ls[0]["x"][:min_len]
    mean = np.mean([sample["y"][:min_len] for sample in ls], axis=0)
    sd = np.std([sample["y"][:min_len] for sample in ls], axis=0)
    return {"x": x, "y":{"mean": mean, "sd": sd}}

def mean_progeny(ls):
    """Runs on all columns"""
    min_len = min([len(sample["time"]) for sample in ls])
    dict_res = dict(x=ls[0]["time"][:min_len])
    
    col = {colname for sample in ls for colname in sample.columns} - {"time"}

    for c in col:
        mean = np.mean([sample.get(c, [0] * min_len)[:min_len] for sample in ls], axis=0)
        sd = np.std([sample.get(c, [0] * min_len)[:min_len] for sample in ls], axis=0)
        dict_res[c] = {"mean": mean, "sd": sd}
    return dict_res

def mean_dict(ls):
    dict_res = dict()
    keys = set().union(*[element.keys() for element in ls])
    for k in keys:
        vec = [element[k] for element in ls if k in element]
        dict_res[k] = {"mean": np.mean(vec), "sd": np.std(vec)}
        
    return dict_res

def merge_parents(df):
    tmp_df = pd.merge(df, df, how='inner', left_on="child1", right_on="index", suffixes=('_M', '_D1'))
    full_df = pd.merge(tmp_df, df.rename(columns=lambda x: x + "_D2"), how='inner', left_on="child2_M",
                       right_on="index_D2", suffixes=('_M', '_D2'))
    full_df["group"] = 2 + 1 * (full_df["appear_time_M"] > 75)
    full_df
    return full_df

def get_sub_df_merged(full_df, key1, key2, group):
    no_gp = (full_df["type_D1"] != "GP") & (full_df["type_D2"] != "GP")
    notnull = full_df[key1].notnull() & full_df[key2].notnull()
    notzero = (full_df[key1] != 0.) & (full_df[key2] != 0.)
    goodgroup = full_df["group"].isin(group)
    filt =  goodgroup & no_gp & notnull & notzero
    cur_df = full_df.loc[filt, :]
    return cur_df

def metrics_tc_mother_daughter(full_df):
    key1, key2 = "Tc_h_M", "Tc_h_D1"
    cur_df = get_sub_df_merged(full_df, key1, key2, [2, 3])
    var1 = cur_df[key1] / cur_df[key2]
    var2 = cur_df[key1] - cur_df[key2]
    return dict(
        mean_ratio_m_d_g23=np.mean(var1), 
        std_ratio_m_d_g23=np.std(var1), 
        mean_diff_m_d_g23=np.mean(var2), 
        std_diff_m_d_g23=np.std(var2)
    )

def metrics_tc_daughters(full_df):
    key1, key2 = "Tc_h_D1", "Tc_h_D2"
    cur_df_g2 = get_sub_df_merged(full_df, key1, key2, [2])
    cur_df_g3 = get_sub_df_merged(full_df, key1, key2, [3])
    cur_df_g23 = get_sub_df_merged(full_df, key1, key2, [2, 3])
    return dict(
        corr_tc_daughter_g2=stats.pearsonr(cur_df_g2[key1], cur_df_g2[key2])[0],
        corr_tc_daughter_g3=stats.pearsonr(cur_df_g3[key1], cur_df_g3[key2])[0],
        corr_tc_daughter_g23=stats.pearsonr(cur_df_g23[key1], cur_df_g23[key2])[0],
    )

def corr_tc_output(full_df):
    no_gp = (full_df["type_D1"] != "GP") & (full_df["type_D2"] != "GP")
    prog_df = full_df[no_gp].copy()
    prog_df["prog_D1"] = prog_df["type_D1"].apply(lambda x: "Cycling" if x in ["IP", "RG"] else "PM")
    prog_df["prog_D2"] = prog_df["type_D2"].apply(lambda x: "Cycling" if x in ["IP", "RG"] else "PM")
    prog_df["nb_child_pm"] = (prog_df["prog_D1"] == "PM").astype(int) + (prog_df["prog_D2"] == "PM").astype(int)
    prog_df_g2 = prog_df[prog_df["group"].isin([2])]
    prog_df_g3 = prog_df[prog_df["group"].isin([3])]
    prog_df_g23 = prog_df[prog_df["group"].isin([2, 3])]
    return dict(
        corr_tc_output_g2=stats.pearsonr(prog_df_g2["nb_child_pm"], prog_df_g2["Tc_h_M"])[0],
        corr_tc_output_g3=stats.pearsonr(prog_df_g3["nb_child_pm"], prog_df_g3["Tc_h_M"])[0],
        corr_tc_output_g23=stats.pearsonr(prog_df_g23["nb_child_pm"], prog_df_g23["Tc_h_M"])[0],
    )

def fate_corr(full_df):
    kC, kN = "Cycling", "PM"
    prog_df = full_df.copy()
    no_gp = (full_df["type_D1"] != "GP") & (full_df["type_D2"] != "GP")
    filt = prog_df["group"].isin([2, 3]) & no_gp
    prog_df = prog_df[filt]
    prog_df["prog_D1"] = prog_df["type_D1"].apply(lambda x: kC if x in ["IP", "RG", "Cycling"] else kN)
    prog_df["prog_D2"] = prog_df["type_D2"].apply(lambda x: kC if x in ["IP", "RG", "Cycling"] else kN)
    res_fate_cor = prog_df.groupby(["prog_D1", "prog_D2"]).size()
    
    CC, CN, NN = res_fate_cor[kC][kC], res_fate_cor[kC][kN] + res_fate_cor[kN][kC], res_fate_cor[kN][kN]
    T = CC + CN + NN
    pCC, pCN, pNN = CC / T, CN / T, NN / T
    all_C, all_N = 2 * CC + CN, 2 * NN + CN
    pC, pN = all_C / (T * 2), all_N / (T * 2)
    eCC, eCN, eNN = pC**2, 2*pC*pN, pN**2
    F_metric = 1 - pCN / eCN
    return dict(
        F_metric=F_metric,
    )

def preprocess_corr_metrics(param, root):
    history = preprocess_history(param, root)
    merged = merge_parents(history)
    res = dict(
        **metrics_tc_mother_daughter(merged),
        **metrics_tc_daughters(merged),
        **corr_tc_output(merged),
        **fate_corr(merged)
    )
    return res