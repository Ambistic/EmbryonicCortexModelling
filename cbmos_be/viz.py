import matplotlib.pyplot as plt
import os
from biodata import plot_number_cells, ratio_eomes

def plot_size(stats, name=None, root="."):
    ref = stats.whole_pop_size.iloc[0]
    plot_number_cells()
    p1 = plt.plot(stats.time, stats.progenitor_pop_size / ref, label="Progenitor population")
    # p2 = plt.plot(stats.time, stats.whole_pop_size / ref, label="Whole Population")
    plt.legend()
    if name:
        plt.savefig(os.path.join(root, f"output/results/prog_size_{name}.png"))
    
def plot_ratio(stats, name=None, root="."):
    stats = stats.fillna(0)
    if "size_type_Cycling" in stats.columns:
        non_IP = stats.size_type_Cycling
        IP = 0
    else:
        non_IP = stats.size_type_RG + stats.size_type_GP if "size_type_GP" in stats.columns else stats.size_type_RG
        IP = stats.size_type_IP
    
    plt.plot(stats.time, IP / (non_IP + IP),
             label="IP ratio")
    plt.plot(ratio_eomes.index, ratio_eomes.val / 100, label="Reference IP ratio")
    plt.legend()
    if name:
        plt.savefig(os.path.join(root, f"output/results/IP_ratio_{name}.png"))
        
def plot_size_ratio(stats, name=None, root="."):
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plot_size(stats)
    plt.subplot(2, 1, 2)
    plot_ratio(stats)
    if name:
        plt.savefig(os.path.join(root, f"output/results/prog_size_ratio_{name}.png"))