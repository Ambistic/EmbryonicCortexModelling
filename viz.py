import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def node_pos(node):
    x = node["dist_N"] / (node["dist_N"] + node["dist_S"] + 1e-6)
    y = node["dist_E"] / (node["dist_E"] + node["dist_W"] + 1e-6)
    x, y = (x - 0.5) * 1.5, (y - 0.5) * 1.5
    return np.array([x, y])

def embedded_viz(gpn, finetune=False):
    # gpn.update_all_dist()
    # pos is a dict of np array with coordinates
    pos = {n: node_pos(gpn.G.nodes[n]) for n in gpn.G.nodes}
    if finetune:
        pos = pos = nx.spring_layout(gpn.G, pos=pos, k=0.01, iterations=1)
    return pos

def quick_export(gpn, pos, name):
    gpn.update_all_dist()
    fig = gpn.show_dist(pos=pos, figsize=(12, 12))
    plt.savefig(name)
    fig.clear()
    plt.close()