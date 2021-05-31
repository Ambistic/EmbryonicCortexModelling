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
        self.debug = False
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

    def debug_dict(self, d):
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
        nx.draw_networkx(
            self.D,
            dual_pos,
            nodelist=nodes,
            edgelist=[e for e in self.D.edges if -1 not in e],
            labels={n: n for n in nodes},
            node_color="lightblue",
        )

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

    def perimeter(self):
        return self.D.degree(-1)

    def density(self):
        return self.G.number_of_nodes() / self.perimeter() ** 2

    def sides(self, node):
        return self.G.nodes[node].get("side", set())

    def ngb(self, node, net="planar"):
        G = self.G if net == "planar" else self.D
        return G.nodes[node]["ngb"]

    def set_ngb(self, node, ngb, net="planar"):
        G = self.G if net == "planar" else self.D
        G.nodes[node]["ngb"] = CircularList(ngb)

    # helpers (to use at the end to simplify the code)
    def dual(self, edge):
        if len(edge) == 2:
            return self.G.edges[edge]["dual"]

        elif len(edge) == 3:
            return self.D.edges[edge]["dual"]

        else:
            raise ValueError(f"edge not recognised : {edge}")

    def set_dual(self, edge, dual):
        if len(edge) == 2 and len(dual) == 3:
            self.G.edges[edge]["dual"] = Ltuple(dual)

        elif len(edge) == 3 and len(dual) == 2:
            self.D.edges[edge]["dual"] = Ltuple(dual)

        else:
            raise ValueError(f"edges not recognised or did not match : {edge}, {dual}")

    def is_border_node(self, node):
        return -1 in self.get_intermediate_neighbours(node)

    def is_border_edge(self, edge):
        return -1 in self.dual(edge)

    def is_corner(self, node):
        return len(self.sides(node)) >= 2

    def corrected_degree(self, node):
        return self.G.degree(node) + len(self.sides(node))


class CheckerPlanarNetwork(PlanarNetwork):
    def check_dual_ngb_consistency(self, node):
        if node not in self.D.nodes:
            return

        ngb_edges = self.D.nodes[node]["ngb"]
        nb_ngbs = len(ngb_edges)
        map_edge_pair = self.planar_cycle_pairs(node)
        ordered_cycle_pairs = cycle_from_ordered_list_pairs(
            [map_edge_pair[x] for x in ngb_edges]
        )
        error = f"Uncorrect pattern in cycle {ordered_cycle_pairs} for node {node}"
        length = len(ordered_cycle_pairs)
        for i in range(1, length):
            self.check(
                ordered_cycle_pairs[i][0] == ordered_cycle_pairs[i - 1][1], error
            )
        self.check(
            ordered_cycle_pairs[0][0] == ordered_cycle_pairs[length - 1][1], error
        )

    def check_planar_ngb_consistency(self, node):
        if node not in self.G.nodes:
            return

        ngb_edges = self.G.nodes[node]["ngb"]
        nb_ngbs = len(ngb_edges)
        map_edge_pair = self.dual_cycle_pairs(node)
        ordered_cycle_pairs = cycle_from_ordered_list_pairs(
            [map_edge_pair[x] for x in ngb_edges]
        )
        error = f"Uncorrect pattern in cycle {ordered_cycle_pairs} for node {node}"
        length = len(ordered_cycle_pairs)
        for i in range(1, length):
            self.check(
                ordered_cycle_pairs[i][0] == ordered_cycle_pairs[i - 1][1], error
            )
        self.check(
            ordered_cycle_pairs[0][0] == ordered_cycle_pairs[length - 1][1], error
        )

    def check_duality_consistency(self):
        for *e, d in self.G.edges(data=True):
            d_e = d["dual"]
            dd_e = self.D.edges[d_e]["dual"]
            try:
                assert Ltuple(dd_e) == Ltuple(e)
            except:
                raise ValueError(f"{dd_e} is not equal to {e} in G graph")

        for *e, d in self.D.edges(data=True, keys=True):
            d_e = d["dual"]
            dd_e = self.G.edges[d_e]["dual"]
            try:
                assert Ltuple(dd_e) == Ltuple(e)
            except:
                raise ValueError(f"{dd_e} is not equal to {e} in D graph")

    def check_ngbs_consistency(self):
        for n in self.G.nodes:
            self.check_planar_ngb_consistency(n)

        for n in self.D.nodes:
            if n == -1:
                continue
            self.check_dual_ngb_consistency(n)

    def check_deep_ngb_consistency(self):
        for n in self.G.nodes:
            cycle = self.get_cycle_pairs(n)
            if len(cycle) == 2:
                continue  # nothing to check more than in check_ngbs_consistency

            for i in range(len(cycle)):
                # for each pair, we must check that the common node has the same pattern
                p1, p2 = cycle[i], cycle[(i + 1) % len(cycle)]
                node = p1[1]  # also p2[0]
                if node == -1:
                    continue
                # exception for -1
                ngb_dual_node = self.ngb(node, net="dual")
                err_msg = (
                    f"Order is not correct for {n} with dual {node}. Cycle is {cycle} "
                    f"and dual ngb is {ngb_dual_node}"
                )
                assert ngb_dual_node.match_pattern([Ltuple(p2), Ltuple(p1)]), err_msg

    def check_void_doublet(self):
        ls = self.ngb(-1, net="dual")
        assert len(ls) == len(set(ls)), f"Doublets found in void ngb, {ls}"

    def check_void_ngbs(self):
        for e in self.D.nodes[-1]["ngb"]:
            assert (
                e in self.D.edges
            ), f"Error edge {e} is in ngbs of -1 but not in graph D"

    def check_all(self):
        self.check_duality_consistency()
        self.check_ngbs_consistency()
        self.check_deep_ngb_consistency()
        self.check_void_doublet()
        self.check_void_ngbs()
