from gpn6 import GrowingPlanarNetwork
import networkx as nx
import random
import numpy as np


seed = 1
random.seed(seed)
np.random.seed(seed)
gpn = GrowingPlanarNetwork()
gpn.init_square(6)

def p_dupl():
    return np.random.random() > 0.48

# plt.figure(figsize=(8, 8))
# nx.draw_networkx(gpn.G, pos=pos)
ratio = 0
for i in range(300):
    # print(gpn.G.number_of_nodes(), ratio)
    # gpn.check_all()
    if p_dupl():
        ratio += 1
        gpn.duplicate_random_node()
    else:
        ratio -= 1
        gpn.remove_random_node()

    if i % 100 == 0:
        print()
        print("Iteration", i, "ratio", ratio, "density", gpn.density())

gpn.check_all()
print("Statistically, everything is fine")
