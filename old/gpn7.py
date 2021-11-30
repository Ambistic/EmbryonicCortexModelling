import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import warnings
from helper import *
from planarnetwork import CheckerPlanarNetwork
import inspect

def gpn_action(func):
    def wrapper(self, *args, **kwargs):
        if self.debug:
            name = func.__name__
            print("Calling", name," with ", args, kwargs)

        res = func(self, *args, **kwargs)
        self.check_all_if_debug()
        return res

    return wrapper


class GrowingPlanarNetwork(CheckerPlanarNetwork):
    """
    Growing and shrinking rules are as follow :
    - ...
    """
    
    def copy(self):
        gpn = super().copy()
        for n in gpn.G.nodes:
            gpn.G.nodes[n]["ngb"] = self.G.nodes[n]["ngb"].copy()
        
        for n in gpn.D.nodes:
            gpn.D.nodes[n]["ngb"] = self.D.nodes[n]["ngb"].copy()
            
        return gpn

    def index_from_coord(self, i, j, size):
        if i < 0 or j < 0 or j >= size or i >= size:
            return None
        return i * size + j

    ###############################
    # ******** INTERFACE ******** #
    ###############################

    def init_tissue(self, size=3):
        self._init_square(size=size)

    def duplicate(self, node, stabilize=True):
        new_node = self._duplicate_node(node)

        if stabilize:
            self.stabilize([node, new_node])

        return new_node

    def destroy(self, node, stabilize=True):
        ngbs = self.ngb(node)
        self._destroy_node(node)

        if stabilize:
            self.stabilize(ngbs)

    def swap(self, node1, node2):
        self._swap_node(node1, node2)

    def stabilize(self, nodes):
        """
        nodes can be a single node or a list of nodes
        """
        self.stabilize_around(nodes)

    def duplicate_random(self):
        n = random.choice(list(self.G.nodes))
        try:
            return self.duplicate(n)
        except:
            print("Error trying to duplicate", n)
            raise

    def destroy_random(self):
        n = random.choice(list(self.G.nodes))
        try:
            return self.destroy(n)
        except:
            print("Error, trying to remove", n)
            raise


    #############################
    # ******** ACTIONS ******** #
    #############################

    """
    Actions are the functions that directly access the network G or D
    by adding or removing edges and nodes, or changing ngbs etc
    """

    # checked for size=3
    @gpn_action
    def _init_square(self, size=3):
        void = -1
        G = nx.Graph()
        D = nx.MultiGraph()
        D.add_node(-1)
        D.nodes[-1]["ngb"] = CircularList()  # could be a void list
        for i in range(size):
            for j in range(size):
                pass

        for i in range(size):
            for j in range(size):
                this = self.index_from_coord(i, j, size)
                less = self.index_from_coord(i - 1, j - 1, size)
                dualless = self.index_from_coord(i - 1, j - 1, size - 1)
                if i < size - 1 and j < size - 1:
                    dualthis = self.index_from_coord(i, j, size - 1)
                    D.add_node(dualthis)
                    D.nodes[dualthis]["ngb"] = CircularList()

                if j > 0:
                    # horizontal planar
                    target = self.index_from_coord(i, j - 1, size)
                    dualtarget = self.index_from_coord(i, j - 1, size - 1)
                    G_edge = (this, target)
                    # horizontal dual
                    if i == 0:
                        D_edge = (void, dualtarget)
                    elif i == (size - 1):
                        D_edge = (void, dualless)
                    else:
                        D_edge = (dualtarget, dualless)
                    idx = D.add_edge(*D_edge, dual=G_edge)
                    G.add_edge(*G_edge, dual=(*D_edge, idx))
                    if D_edge[0] is not None:
                        D.nodes[D_edge[0]]["ngb"].append(Ltuple((*D_edge, idx)))
                    if D_edge[1] is not None:
                        D.nodes[D_edge[1]]["ngb"].append(Ltuple((*D_edge, idx)))

                if i > 0:
                    # vertical planar
                    target = self.index_from_coord(i - 1, j, size)
                    dualtarget = self.index_from_coord(i - 1, j, size - 1)
                    G_edge = (this, target)
                    # vertical dual
                    if j == 0:
                        D_edge = (void, dualtarget)
                    elif j == (size - 1):
                        D_edge = (void, dualless)
                    else:
                        D_edge = (dualtarget, dualless)

                    idx = D.add_edge(*D_edge, dual=G_edge)
                    G.add_edge(*G_edge, dual=(*D_edge, idx))
                    if D_edge[0] is not None:
                        D.nodes[D_edge[0]]["ngb"].append(Ltuple((*D_edge, idx)))
                    if D_edge[1] is not None:
                        D.nodes[D_edge[1]]["ngb"].append(Ltuple((*D_edge, idx)))

        # set cyclic values for all nodes
        ngbs = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        for i in range(size):
            for j in range(size):
                this = self.index_from_coord(i, j, size)
                # planar
                G.nodes[this]["ngb"] = CircularList()
                for ngb in ngbs:
                    # TODO get edge instead of node
                    new = self.index_from_coord(*add((i, j), ngb), size)
                    if new is not None:
                        G.nodes[this]["ngb"].append(new)

        self.G = G
        self.D = D

    @gpn_action
    def _duplicate_node(self, node):
        """
        High level function
        We have a graph, a dual graph, edge indexes and ngb cyclic order
        All must stay ok to prove consistence through recurrence

        Division step:
        A1) Split neighbours according to the cyclic order
        A1b) Construct "neighbour" dual node (intermediate graph with order)
        A2) Update edges and dual graph edges accordingly
        A3) Add inner edge with corresponding dual

        Compensation step:
        B1) Get previous neighbour dual node
        B2) Select dual node if #ngb > 4
        B3) For all of them, iteratively
        B4) Select pair of non-adjacent nodes with lowest degree
        B5) As for normal step, run a split neighbour + edge update + inner edge

        Make graphics to help understand
        """

        # A1
        ngbs = self.ngb(node)
        nb_ngbs = len(ngbs)
        half_1 = nb_ngbs // 2
        half_2 = nb_ngbs - half_1

        split_ngb = self.pick_best_split(node, ngbs)

        # A1b
        cycle_dual = self.dual_cycle_pairs(node)

        # A2 remove edge
        for ngb in ngbs:
            self.G.remove_edge(node, ngb)

        # division
        new_node = self.index("planar")
        self.G.add_node(new_node)
        current = split_ngb

        self.G.nodes[node]["ngb"] = CircularList()
        self.G.nodes[new_node]["ngb"] = CircularList()

        for i in range(half_1):
            self.G.add_edge(node, current, dual=cycle_dual[current])
            self.G.nodes[node]["ngb"].append(current)
            current = ngbs.next(current)
        self.G.nodes[node]["ngb"].append(new_node)

        opp_split_ngb = current
        for j in range(half_2):
            self.G.add_edge(new_node, current, dual=cycle_dual[current])
            self.D.edges[cycle_dual[current]]["dual"] = (new_node, current)
            self.G.nodes[new_node]["ngb"].append(current)
            # update ngb
            ls_to_update = self.G.nodes[current]["ngb"]
            ls_to_update[ls_to_update.index(node)] = new_node
            current = ngbs.next(current)
        self.G.nodes[new_node]["ngb"].append(node)

        # A3
        first_dual_edge = Ltuple(cycle_dual[split_ngb])
        second_dual_edge = Ltuple(cycle_dual[opp_split_ngb])

        if len(cycle_dual) >= 3:
            # implicit, nope it is not ok at all
            first_dual, second_dual = first_dual_edge[0], second_dual_edge[0]

        elif len(cycle_dual) == 2:
            assert min(first_dual_edge[:2]) == min(second_dual_edge[:2])
            assert max(first_dual_edge[:2]) == max(second_dual_edge[:2])
            # need to use the non -1 dual as an anchor
            first_dual, second_dual = max(first_dual_edge[:2]), min(first_dual_edge[:2])
            ref_order = self.D.nodes[first_dual]["ngb"]
            if ref_order.match_pattern([first_dual_edge, second_dual_edge]):
                pass

            elif ref_order.match_pattern([second_dual_edge, first_dual_edge]):
                first_dual_edge, second_dual_edge = second_dual_edge, first_dual_edge

            else:
                raise RuntimeError("pattern matching on dual edge did not work")

        idx = self.D.add_edge(first_dual, second_dual, dual=(node, new_node))
        inner_dual_edge = Ltuple((first_dual, second_dual, idx))
        self.G.add_edge(node, new_node, dual=inner_dual_edge)

        # keep order for dual inner
        # to do so, cycle_dual has information on the dual edges
        # so we must insert it after this one
        idx_1 = index_of(first_dual_edge, self.D.nodes[first_dual]["ngb"])
        self.D.nodes[first_dual]["ngb"].insert(idx_1 + 1, inner_dual_edge)
        idx_2 = index_of(second_dual_edge, self.D.nodes[second_dual]["ngb"])
        self.D.nodes[second_dual]["ngb"].insert(idx_2 + 1, inner_dual_edge)

        # B
        # stabilize cycle
        # dual_list = list_from_cycle_dual(cycle_dual)
        # for dual_node in dual_list:
        #     self.shorten_cycle(dual_node)

        return new_node

    @gpn_action
    def _destroy_node(self, node):
        """
        In a remove node, we first connect all neighbours that are not
        A) for all ngb pairs, add edge if not exists
        B) effectively remove the central node
        C) shorten the central cycle

        Returns node neighbours for stabilization
        """
        # A
        # A1
        ngbs = self.ngb(node)
        nb_ngbs = len(ngbs)

        # A1b
        cycle_dual = self.dual_cycle_pairs(node)

        # could just put number of ngbs ...
        if len(cycle_dual) == 2:
            # case where node is just removed
            if (ngbs[0], ngbs[1]) in self.G.edges:
                self._remove_edge((ngbs[0], ngbs[1]))

            self._quick_remove(node)
            return

        # create connections between its neighbours
        for i in range(nb_ngbs):
            n_a, n_b = ngbs[i], ngbs[(i + 1) % nb_ngbs]

            if not (n_a, n_b) in self.G.edges:
                ref_dual_node = self.common_dual((node, n_a), (node, n_b))
                self._create_edge(n_a, n_b, ref_dual_node=ref_dual_node, ref_node=node)

            # if edge but dual is not correct
            elif not self.are_from_common_cycle(node, n_a, n_b):
                self._remove_edge((n_a, n_b))
                ref_dual_node = self.common_dual((node, n_a), (node, n_b))
                self._create_edge(n_a, n_b, ref_dual_node=ref_dual_node, ref_node=node)

        # remove the node and merge dual neighbours
        self._remove_node(node)

    @gpn_action
    def _swap_node(self, node1, node2):
        """
        We could only do a reference swap, however this makes harder the monitoring.
        Therefore we implement it
        """
        # sanity check
        assert (
            node1,
            node2,
        ) in self.G.edges, f"Trying to swap non-adjacent nodes {node1} and {node2}"

        ngb1, ngb2 = self.ngb(node1), self.ngb(node2)
        ngb1[ngb1.index(node2)] = node1
        ngb2[ngb2.index(node1)] = node2
        self.set_ngb(node2, ngb1)
        self.set_ngb(node1, ngb2)

        def ngb_update(self, nodeA, nodeB, ngbs):
            for ngb in ngbs:
                if ngb == nodeA:
                    continue
                dual_edge = self.dual((nodeA, ngb))
                self.set_dual(dual_edge, (nodeB, ngb))
                self.G.add_edge(nodeB, ngb, dual=dual_edge)
                self.G.remove_edge(nodeA, ngb)
                self.ngb(ngb).replace(nodeA, nodeB)

        ngb_update(self, node1, node2, ngb1)
        ngb_update(self, node2, node1, ngb2)

    ###############################
    # ******** LOW LEVEL ******** #
    ###############################

    def _make_edge(self, dual_node, min_pair):
        # B5
        if self.verbose:
            print("Shorten", min_pair, "on", dual_node)

        Z = self.D.nodes[-1]["ngb"]

        # A1
        ngb_edges = self.D.nodes[dual_node]["ngb"]
        nb_ngbs = len(ngb_edges)

        # A1b
        map_edge_pair = self.planar_cycle_pairs(dual_node)

        ordered_cycle_pairs = cycle_from_ordered_list_pairs(
            [map_edge_pair[x] for x in ngb_edges]
        )

        ingbs = CircularList([x[0] for x in ordered_cycle_pairs])

        map_ingb_pair = {x[0]: x for x in ordered_cycle_pairs}

        # A2 remove edge
        for edge in ngb_edges:
            self.D.remove_edge(*edge)

        # division
        new_node = self.index("dual")
        self.D.add_node(new_node)

        split_ngb_1, split_ngb_2 = min_pair
        current = split_ngb_1
        c_idx = ingbs.index(current)

        self.D.nodes[dual_node]["ngb"] = CircularList()
        self.D.nodes[new_node]["ngb"] = CircularList()

        # assert len(set(ingbs)) == len(ingbs), (
        #     f"Error found pair in {ingbs} with "
        #     f"{map_edge_pair} and {ordered_cycle_pairs}"
        # )

        assert split_ngb_2 in ingbs, (
            f"Error couldn't find {split_ngb_2} in {ingbs} with "
            f"{map_edge_pair} and {ordered_cycle_pairs}"
        )

        while current != split_ngb_2:
            planar_edge = map_ingb_pair[current]
            dual_edge = self.dual(planar_edge)

            # here id is not taken into consideration
            self.D.add_edge(*dual_edge, dual=planar_edge)
            self.D.nodes[dual_node]["ngb"].append(dual_edge)

            c_idx = ingbs.next_id(c_idx)
            current = ingbs[c_idx]
            # current = ingbs.next(current)

        assert split_ngb_1 in ingbs, (
            f"Error couldn't find {split_ngb_2} in {ingbs} with "
            f"{map_edge_pair} and {ordered_cycle_pairs}"
        )
        while current != split_ngb_1:
            planar_edge = map_ingb_pair[current]
            dual_edge = self.dual(planar_edge)

            try:
                current_edge = self.replace_node(dual_edge, dual_node, new_node)[:2]
            except:
                print(locals())
                raise

            idx = self.D.add_edge(*current_edge, dual=planar_edge)
            new_edge = (*current_edge, idx)

            self.G.edges[planar_edge]["dual"] = new_edge
            self.D.nodes[new_node]["ngb"].append(new_edge)

            dual_current = self.get_other_node(dual_edge, dual_node)
            ls_to_update = self.D.nodes[dual_current]["ngb"]

            try:
                ls_to_update[index_of(dual_edge, ls_to_update)] = new_edge
            except:
                print(locals())
                print(self.D.nodes[-1]["ngb"], Z)
                raise

            c_idx = ingbs.next_id(c_idx)
            current = ingbs[c_idx]
            # current = ingbs.next(current)

        idx = self.D.add_edge(dual_node, new_node)
        inner_dual_edge = Ltuple((dual_node, new_node, idx))
        self.D.nodes[dual_node]["ngb"].append(inner_dual_edge)
        self.D.nodes[new_node]["ngb"].append(inner_dual_edge)

        self.G.add_edge(split_ngb_1, split_ngb_2, dual=inner_dual_edge)
        self.D.edges[inner_dual_edge]["dual"] = (split_ngb_1, split_ngb_2)

        idx_1 = index_of(
            map_ingb_pair[split_ngb_1][1], self.G.nodes[split_ngb_1]["ngb"]
        )
        self.G.nodes[split_ngb_1]["ngb"].insert(idx_1 + 1, split_ngb_2)
        idx_2 = index_of(
            map_ingb_pair[split_ngb_2][1], self.G.nodes[split_ngb_2]["ngb"]
        )
        self.G.nodes[split_ngb_2]["ngb"].insert(idx_2 + 1, split_ngb_1)

        return new_node

    def _remove_node(self, node):
        ingbs = self.get_intermediate_neighbours(node, net="planar")
        ngbs = self.G.nodes[node]["ngb"]

        # DON'T UNDERSTAND
        # create a function that reconstruct the future "ngb" of the merged node
        new_ingbs, old_edges = self._get_merged_neighbours(node, ingbs)

        for ngb in ngbs:
            self.G.nodes[ngb]["ngb"].remove(node)

        self.G.remove_node(node)

        # create this new merged node, remove others and change names in other "ngb"
        # get the minus to pick the -1 if there
        keeped_dual = min(ingbs)

        # add edges to get their id
        keeped_dual_new_ngb = CircularList()

        for new_ingb, old_edge in zip(new_ingbs, old_edges):
            idx = self.D.add_edge(keeped_dual, new_ingb)
            new_edge = (keeped_dual, new_ingb, idx)
            keeped_dual_new_ngb.append(new_edge)

            # dual edge update
            dual_edge = self.D.edges[old_edge]["dual"]
            self.D.edges[new_edge]["dual"] = dual_edge
            self.G.edges[dual_edge]["dual"] = new_edge

            # replace in "ngb"
            self.D.nodes[new_ingb]["ngb"].replace(Ltuple(old_edge), Ltuple(new_edge))

            # remove old edge
            self.D.remove_edge(*old_edge)

        if keeped_dual == -1:
            # add new edges to -1
            self.D.nodes[keeped_dual]["ngb"] += keeped_dual_new_ngb
            # remove edges with nodes that does not exist anymore
            self._remove_edges_with(keeped_dual, ingbs)
        else:
            self.D.nodes[keeped_dual]["ngb"] = keeped_dual_new_ngb

        for node in ingbs:
            if node != keeped_dual:
                self.D.remove_node(node)

    def _remove_edge(self, edge):
        """
        1 pick dual edge
        2 rebuild ngb from both ngb and next (implicitly remove the old dual)
        3 remove from planar ngb
        4 remove + readd dual edges
        5 update duality
        6 replace in the ngb of others
        X might be ok
        TODO here -1 ngb is disrupted
        """
        # 1
        dual_edge = Ltuple(self.dual(edge))
        dual_0, dual_1, _ = dual_edge
        if dual_1 == -1:
            dual_0, dual_1 = dual_1, dual_0

        # 2
        ngb_0 = self.ngb(dual_0, net="dual")
        ngb_1 = self.ngb(dual_1, net="dual")
        # ls_0 = [] if (dual_0 == -1) else self.cycle_ngb(ngb_0, from_=dual_edge, to_=dual_edge)
        ls_0 = self.cycle_ngb(ngb_0, from_=dual_edge, to_=dual_edge)
        ls_1 = self.cycle_ngb(ngb_1, from_=dual_edge, to_=dual_edge)

        # 3
        self.G.remove_edge(*edge)

        # 4
        ls_1_true = []
        for i_dual_edge in ls_1:
            i_dual_node = self.get_other_node(i_dual_edge, dual_1)
            i_planar_edge = self.dual(i_dual_edge)
            self.D.remove_edge(*i_dual_edge)
            idx = self.D.add_edge(dual_0, i_dual_node)
            i_new_dual_edge = Ltuple((dual_0, i_dual_node, idx))
            self.set_cross_dual(i_new_dual_edge, i_planar_edge)
            self.ngb(i_dual_node, net="dual").replace(
                Ltuple(i_dual_edge), i_new_dual_edge
            )
            ls_1_true.append(i_new_dual_edge)

        # 5
        self.D.remove_node(dual_1)
        new_dual_ngb = CircularList(ls_0 + ls_1_true)
        self.set_ngb(dual_0, new_dual_ngb, net="dual")

        # 6
        self.ngb(edge[0]).remove(edge[1])
        self.ngb(edge[1]).remove(edge[0])

        return dual_0

    def _create_edge(self, source, target, ref_node=None, ref_dual_node=None):
        """
        Only for planar graph, not for dual
        """
        graph_source = self.get_intermediate_neighbours(source, net="planar")
        graph_target = self.get_intermediate_neighbours(target, net="planar")
        inter = list(set(graph_source) & set(graph_target))

        if len(inter) > 1:
            if ref_dual_node is not None:
                dual_node = ref_dual_node
            else:
                raise RuntimeError(
                    "ref_dual_node shall not be None" "when there is an ambiguity"
                )

        else:
            dual_node = inter[0]

        if dual_node == -1:
            self._make_border_edge(ref_node, (source, target))
        else:
            self._make_edge(dual_node, (source, target))

    def _quick_remove(self, node):
        assert (
            self.G.degree(node) == 2
        ), f"Illegal call of _quick_remove for node {node}"
        """
        Things to do are :
        -1 create a naked edge between neighbours
        -2 remove a random dual edge among the two available
        -3 remove old planar edges
        -4 remove node
        -5 reset duality in naked planar edge and keeped dual edge
        -6 update planar ngb (replace)
        -7 update dual ngb (remove)
        """
        # 1
        ngbs = self.ngb(node)
        self.G.add_edge(ngbs[0], ngbs[1])

        # 2
        keeped_dual_edge = self.dual((node, ngbs[0]))
        removed_dual_edge = self.dual((node, ngbs[1]))
        self.D.remove_edge(*removed_dual_edge)

        # 3
        self.G.remove_edge(node, ngbs[0])
        self.G.remove_edge(node, ngbs[1])

        # 4
        self.G.remove_node(node)

        # 5
        self.set_dual((ngbs[0], ngbs[1]), keeped_dual_edge)
        self.set_dual(keeped_dual_edge, (ngbs[0], ngbs[1]))

        # 6
        self.ngb(ngbs[0]).replace(node, ngbs[1])
        self.ngb(ngbs[1]).replace(node, ngbs[0])

        # 7
        self.ngb(removed_dual_edge[0], net="dual").remove(removed_dual_edge)
        self.ngb(removed_dual_edge[1], net="dual").remove(removed_dual_edge)

    def _make_border_edge(self, node, min_pair):
        dual_node = -1
        # B5
        if self.verbose:
            print("Shorten", min_pair, "on", dual_node)

        # if number of ngb is 2, then finding pattern does not make sens
        # therefore, we need to find an "anchor" somewhere

        # A2 remove edge
        crossing_pairs = [(node, min_pair[0]), (node, min_pair[1])]
        self.D.remove_edge(*self.dual(crossing_pairs[0]))
        self.D.remove_edge(*self.dual(crossing_pairs[1]))

        # division
        new_node = self.index("dual")
        self.D.add_node(new_node)

        split_ngb_1, split_ngb_2 = min_pair

        # self.D.nodes[dual_node]["ngb"] = CircularList()
        self.D.nodes[new_node]["ngb"] = CircularList()

        # set up correct order
        ngb = self.ngb(node)
        if len(self.ngb(node)) == 2:
            # then we need the anchor
            if not self.outside_with_anchor(node, split_ngb_1, split_ngb_2):
                crossing_pairs = crossing_pairs[::-1]  # exactly the same as below
                split_ngb_1, split_ngb_2 = split_ngb_2, split_ngb_1

        elif ngb.match_pattern(min_pair):
            pass
            # crossing_pairs = crossing_pairs[::-1]  # exactly the same as below
            # split_ngb_1, split_ngb_2 = split_ngb_2, split_ngb_1

        elif ngb.match_pattern(min_pair[::-1]):
            crossing_pairs = crossing_pairs[::-1]  # exactly the same as below
            split_ngb_1, split_ngb_2 = split_ngb_2, split_ngb_1

        else:
            raise ValueError(f"No match for {ngb} and {crossing_pairs}")

        for planar_edge in crossing_pairs:
            dual_edge = self.dual(planar_edge)

            current_edge = self.replace_node(dual_edge, dual_node, new_node)[:2]

            idx = self.D.add_edge(*current_edge, dual=planar_edge)
            new_edge = (*current_edge, idx)

            self.G.edges[planar_edge]["dual"] = new_edge
            self.D.nodes[new_node]["ngb"].append(new_edge)

            dual_current = self.get_other_node(dual_edge, dual_node)
            ls_to_update = self.D.nodes[dual_current]["ngb"]

            ls_to_update[index_of(dual_edge, ls_to_update)] = new_edge

            self.ngb(dual_node, net="dual").remove(Ltuple(dual_edge))

        idx = self.D.add_edge(dual_node, new_node)
        inner_dual_edge = Ltuple((dual_node, new_node, idx))

        # TODO THERE
        self.ngb(dual_node, net="dual").append(inner_dual_edge)
        # rationale is that this node will have 3 ngb and inner comes in 2nd
        # thank to match_pattern
        self.ngb(new_node, net="dual").insert(1, inner_dual_edge)  # not ok
        # self.ngb(new_node, net="dual").append(inner_dual_edge)  # not ok
        # need to define the right order in the for loop THERE

        self.G.add_edge(split_ngb_1, split_ngb_2, dual=inner_dual_edge)
        self.set_dual(inner_dual_edge, (split_ngb_1, split_ngb_2))

        # knowing the right order, there will be one next and one previous THERE
        # I think with the match pattern above, it might be ok
        idx_1 = index_of(node, self.ngb(split_ngb_1))
        self.ngb(split_ngb_1).insert(idx_1, split_ngb_2)  # to check
        idx_2 = index_of(node, self.ngb(split_ngb_2))
        self.ngb(split_ngb_2).insert(idx_2 + 1, split_ngb_1)  # to check

        return new_node

    #############################
    # ******** HELPERS ******** #
    #############################

    def get_intermediate_neighbours(self, node, net="planar"):
        """
        net is either 'planar' or 'dual'
        """
        cycle_pairs = self.get_cycle_pairs(node, net=net)
        return [x[0] for x in cycle_pairs]

    def cycle_ngb(self, circlist, from_, to_):
        """
        Returns a list from `from_` to `to_` both excluded
        """
        cur = from_
        ls = []
        while True:
            cur = circlist.next(cur)
            if cur == from_ or cur == to_:
                break
            ls.append(cur)

        return ls

    def dual_cycle_pairs(self, node):
        ngbs = self.G.nodes[node]["ngb"]
        cycle = dict()  # order is kept by ngbs
        ls_pairs = list()
        for ngb in ngbs:
            ls_pairs.append(self.G.edges[(node, ngb)]["dual"])
        ls_pairs = cycle_from_ordered_list_pairs(ls_pairs)
        return {ngb: pair for ngb, pair in zip(ngbs, ls_pairs)}

    def planar_cycle_pairs(self, node):
        ngbs = self.D.nodes[node]["ngb"]
        cycle = dict()  # order is kept by ngbs
        ls_pairs = list()
        for ngb in ngbs:
            ls_pairs.append(self.D.edges[ngb]["dual"])
        ls_pairs = cycle_from_ordered_list_pairs(ls_pairs)
        return {ngb: pair for ngb, pair in zip(ngbs, ls_pairs)}

    def set_cross_dual(self, edge_1, edge_2):
        self.set_dual(edge_1, edge_2)
        self.set_dual(edge_2, edge_1)

    def are_from_common_cycle(self, a, b, c):
        """
        Only for triangle cycles, otherwise it will likely raise an error
        because of unexisting edge
        """
        common_dual = (
            set(self.dual((a, b))[:2])
            & set(self.dual((a, c))[:2])
            & set(self.dual((b, c))[:2]) - {-1}
        )

        return len(common_dual) == 1

    def get_border_neighbours(self, node):
        ngbs = self.ngb(node)
        for ngb in ngbs:  # this is somehow a match pattern
            next_ngb = ngbs.next(ngb)
            if self.is_border_edge((node, ngb)) and self.is_border_edge(
                (node, next_ngb)
            ):
                return (ngb, next_ngb)

        raise RuntimeError(
            f"Pattern of to succesive border edges were not found node {node}, ngbs {ngbs}"
        )

    def pick_best_split(self, node, ngbs):
        if not self.is_border_node(node):
            return random.choice(ngbs)

        threshold = 1 / 16  # density for rectangular shape

        if self.density() < threshold:
            # print("Correct for density")
            return random.choice(ngbs)

        # returns the seconds such that the other is in the other half
        return self.get_border_neighbours(node)[1]

    def pick_best_pair(self, dual_node, planar_ngb):
        random.shuffle(planar_ngb)
        min_pair = None
        min_score = 1e6
        for i in planar_ngb:
            for j in planar_ngb:
                if j <= i or (i, j) in self.G.edges:
                    continue
                if self.is_border_node(i) and self.is_border_node(j):
                    continue
                score = self.G.degree(i) + self.G.degree(j)
                if score < min_score:
                    min_pair = (i, j)
                    min_score = score

        return min_pair

    def replace_node(self, edge, old, new):
        """
        This function is used to modify an edge in the "ngb" slot

        Example:
        We have (1, 4, 0) and we need to change the 4 by a 5
        However this function is not correct for dual edge
        """
        edge_ = list(edge)
        edge_[edge_.index(old)] = new
        return tuple(edge_)

    def get_other_node(self, edge, node):
        if edge[0] == node:
            return edge[1]

        elif edge[1] == node:
            return edge[0]

        else:
            raise ValueError(f"{node} is not in {edge}")

    def common_dual_old(self, edge_1, edge_2):
        x = set(self.dual(edge_1)[:2]) & set(self.dual(edge_2)[:2]) - {-1}
        if len(x) > 1:
            raise ValueError(
                f"Found multiple common dual (non -1) "
                f"{x} for edges {edge_1} and {edge_2}"
            )
        elif len(x) == 0:
            return None  # because -1 is in common normally

        return list(x)[0]

    def common_dual(self, edge_1, edge_2):
        """
        Returns the commons dual between two edges including -1
        """
        x = set(self.dual(edge_1)[:2]) & set(self.dual(edge_2)[:2])
        if len(x) > 1:
            raise ValueError(
                f"Found multiple common dual" f"{x} for edges {edge_1} and {edge_2}"
            )

        return list(x)[0]

    def outside_with_anchor(self, node, ngb1, ngb2):
        print("Calling outside_with_anchor")
        current = ngb2
        prev = node
        while len(self.ngb(current)) == 2 and current != node:
            current, prev = self.get_other_node(self.ngb(current), prev), current

        if current == node:
            return True

        # assertion
        if len(self.ngb(current)) <= 2:
            raise RuntimeError(
                f"{current} has not enough neighbours, requires at least 3"
            )

        # pick the "next" node
        next_ = self.ngb(current).next(prev)

        print("debug_anchor", locals())

        if self.is_border_node(next_) and (current, next_) in self.G.edges:
            return self.is_border_edge((current, next_))

        return False

    def _remove_edges_with(self, node, ngbs):
        """
        Only for dual graph for now
        """
        ngbs = set(ngbs) - {node}
        new_list = CircularList()
        for ngb in self.D.nodes[node]["ngb"]:
            if ngb[0] in ngbs or ngb[1] in ngbs:
                continue
            new_list.append(ngb)

        self.D.nodes[node]["ngb"] = new_list

    def _get_merged_neighbours(self, node, ingbs):
        # sanity check
        for ingb in ingbs:
            deg = self.D.degree(ingb)
            assert (
                deg == 3 or ingb == -1
            ), f"error for degree {deg}, node {node}, ingb {ingb}"

        new_ingbs = CircularList([])
        old_edges = CircularList([])

        def get_stranger(raw_ngbs, ref_ngb):
            for x in raw_ngbs:
                for y in x[:2]:
                    if y not in ref_ngb:
                        return y, x
            raise RuntimeError("Unable to find another node")

        for ingb in ingbs:
            if ingb == -1:
                continue
            raw_ngbs = self.D.nodes[ingb]["ngb"]
            stranger, edge = get_stranger(raw_ngbs, ingbs)
            new_ingbs.append(stranger)
            old_edges.append(edge)

        return new_ingbs, old_edges

    def get_cycle_pairs(self, node, net="planar"):
        G = self.G if net == "planar" else self.D
        ngb_edges = G.nodes[node]["ngb"]
        nb_ngbs = len(ngb_edges)

        if net == "planar":
            map_edge_pair = self.dual_cycle_pairs(node)
        else:
            map_edge_pair = self.planar_cycle_pairs(node)

        ordered_cycle_pairs = cycle_from_ordered_list_pairs(
            [map_edge_pair[x] for x in ngb_edges]
        )
        return ordered_cycle_pairs

    #################################
    # ******** STABILIZERS ******** #
    #################################

    # TODO this one shall take care of its size, not callers
    def shorten_cycle(
        self, dual_node, source=None, target=None
    ):  # aka Compensation Step
        """
        This function adds an edge between two opposite nodes in a cycle
        if the cycle is longer than 5, because it is not biologically plausible

        Compensation step:
        B1) Get previous neighbour dual node
        B2) Select dual node if #ngb > 4
        B3) For all of them, iteratively
        B4) Select pair of non-adjacent nodes with lowest degree
        B5) As for normal step, run a split neighbour + edge update + inner edge
        """
        if not (self.D.degree(dual_node) > 4 and dual_node != -1):
            return

        planar_ngb = self.get_intermediate_neighbours(dual_node, net="dual")

        min_pair = self.pick_best_pair(dual_node, planar_ngb)

        if not min_pair:
            return

        try:
            return self._make_edge(dual_node, min_pair)

        except:
            print(dual_node, min_pair, "are guilty")
            raise
            
    def stabilize_nodes(self, nodes):
        if isinstance(nodes, int):
            nodes = [nodes]

        fullset = set()
        for n in nodes:
            fullset |= {n}

        for node in fullset:
            self.stabilize_ngb(node)

    def stabilize_around(self, nodes):
        if isinstance(nodes, int):
            nodes = [nodes]

        fullset = set()
        for n in nodes:
            fullset |= {n}
            fullset |= set(self.ngb(n))

        for node in fullset:
            self.stabilize_ngb(node)

    def stabilize_ngb(self, node):
        ngbs = self.ngb(node)
        sides = self.sides(node)

        # MAX NGB
        if len(ngbs) + len(sides) > 5:  # at least 6 neighbours
            # pick one node
            # TODO create a pick_bestfunction for that
            ls = list(map(lambda x: len(self.ngb(x)), ngbs))
            other_node = ngbs[ls.index(max(ls))]
            edge = (other_node, node)

            if -1 in self.dual(edge):
                self.debug_dict({"Warn": "Stabilization tried to occur on border edge"})
                return  # avoid removing border edge

            self._remove_edge(edge)

            # shorten cycle if required
            cycle_dual = self.dual_cycle_pairs(node)
            dual_list = list_from_cycle_dual(cycle_dual)
            for dual_node in dual_list:
                self.shorten_cycle(dual_node)

        # MIN NGB
        if len(ngbs) + len(sides) < 3:
            print(f"Quasi lonely node {node} with #ngbs =", len(ngbs))

        # CROSSING BORDER
        if self.is_border_node(node):
            for ngb in ngbs:
                if self.is_border_node(ngb) and not self.is_border_edge((node, ngb)):
                    print(
                        "Removing crossing border edge",
                        node,
                        ngb,
                        self.dual((node, ngb)),
                    )
                    self._remove_edge((node, ngb))
