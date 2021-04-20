import networkx as nx
from helper import *

class BaseNetwork:
    def check(self, seq, error):
        if seq is not True:
            raise ValueError(error)


class PlanarNetwork(BaseNetwork):
    def __init__(self):
        self.G = nx.Graph()  # planar graph
        self.D = nx.MultiGraph()  # dual graph
        self.ctrl = True
        self.verbose = False
        self.__next_id_D = 0
        self.__next_id_G = 0
        
    def copy(self):
        gpn = GrowingPlanarNetwork()
        gpn.G = self.G.copy()
        gpn.D = self.D.copy()
        gpn.ctrl = self.ctrl
        return gpn
    
    def index(self, net):
        G = self.G if net == "planar" else self.D
        return max(G.nodes) + 1
        if net == "planar":
            if self.__next_id_G <= max(self.G.nodes):
                self.__next_id_G = max(self.G.nodes) + 1
            x = self.__next_id_G
            self.__next_id_G += 1
            return x
        
        elif net == "dual":
            if self.__next_id_D <= max(self.D.nodes):
                self.__next_id_D = max(self.D.nodes) + 1
            x = self.__next_id_D
            self.__next_id_D += 1
            return x
    
    def debug(self, d):
        if self.verbose:
            for k, v in d.items():
                print(k, ":", v)
                
    def show(self, k=5, iterations=1000, dual=False):
        G = self.D if dual else self.G
        pos = nx.spring_layout(G, k=k, iterations=iterations)
        nx.draw_networkx(G, pos, node_color="orange")
        # nx.draw_networkx(G, pos, nodelist=self.get_non_border_nodes(), node_color="lightblue")
        nx.draw_networkx_labels(G, pos)
        """
        To show both planar and dual on the same plot,
        we should compute
        """
                    
    def show_all(self, k=5, iterations=1000, figsize=(10, 10)):
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.G, k=k, iterations=iterations)
        nx.draw_networkx(self.G, pos, node_color="orange")
        
        dual_pos = dict()
        for dn in self.D.nodes:
            if dn == -1:
                continue
            ngb = self.get_intermediate_neighbours(dn, net="dual")
            arr = np.array([pos[x] for x in ngb])
            dual_pos[dn] = arr.mean(0)
            
        nodes = list(dual_pos.keys())
        nx.draw_networkx(self.D, dual_pos, nodelist=nodes,
                         edgelist=[e for e in self.D.edges if -1 not in e],
                         labels={n: n for n in nodes}, node_color="lightblue")
        
    def print_edges(self, dual=False):
        G = self.D if dual else self.G
        for e in G.edges:
            print(e, G.edges[e])
            
    def print_nodes(self, dual=False):
        G = self.D if dual else self.G
        for n in G.nodes:
            print(n, G.nodes[n])
        
    def debug_all(self):
        plt.figure()
        self.show(dual=False)
        plt.figure()
        self.show(dual=True)
        self.print_edges(dual=False)
        self.print_nodes(dual=False)
        self.print_edges(dual=True)
        self.print_nodes(dual=True)