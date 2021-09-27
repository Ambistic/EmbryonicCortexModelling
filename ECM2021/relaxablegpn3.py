from gpn8 import GrowingPlanarNetwork, gpn_action
from helper import *
import networkx as nx
import random
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import gc

def std4(distr):
    distr = np.asarray(distr)
    mean = np.sqrt(len(distr))
    dev = (distr - mean)**4
    return np.mean(dev)

def std4_old(distr):
    distr = np.asarray(distr)
    mean = np.mean(distr)
    dev = (distr - mean)**4
    return np.mean(dev)

class EnergyPoint:
    def __init__(self, node, axis, val):
        self.node = node
        self.axis = axis
        self.val = val
        
    def to_shorten(self):
        return self.val > 1. and self.val < 1e2
    
    def get_norm_val(self):
        if self.val > 1:
            return 1 / self.val
        return self.val
        
    def __gt__(self, other):
        return self.get_norm_val() > other.get_norm_val()
    
    def __lt__(self, other):
        return self.get_norm_val() < other.get_norm_val()
    
    def __eq__(self, other):
        return self.get_norm_val() == other.get_norm_val()
    
    def __repr__(self):
        return f"EnergyPoint => Node : {self.node}, Axis : {self.axis}, Val : {self.val}"
    
class RelaxAction:
    """
    Handles :
    
    """
    def __init__(self, name, args, pattern):
        self.name = name
        self.args = args
        self.pattern = pattern
        self.score = None
        
    def __gt__(self, other):
        return self.score > other.score
    
    def __lt__(self, other):
        return self.score < other.score
    
    def __eq__(self, other):
        return self.score == other.score
        
    def more(self, node):
        nodes = set()
        for sign, edge in self.pattern:
            if sign == "+" and node in edge:
                nodes |= set(edge)
        nodes -= {node}
        return nodes
    
    def less(self, node):
        nodes = set()
        for sign, edge in self.pattern:
            if sign == "-" and node in edge:
                nodes |= set(edge)
        nodes -= {node}
        return nodes
    
    def get_nodes(self):
        nodes = set()
        for _, edge in self.pattern:
            nodes |= set(edge)
            
        return nodes
    
    def __repr__(self):
        return f"RelaxAction : {self.name} with {self.args}, pattern is {self.pattern} : {self.score}"
    
class RelaxableGPN(GrowingPlanarNetwork):
    def __init__(self, *args, callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_action = None
        self.trials = list()
        self.counter = 0
        self.quick = False
        self._callback = callback
        
    def copy(self):
        gpn = super().copy()
        gpn._callback = self._callback
        gpn.counter = self.counter
        gpn.quick = self.quick
            
        return gpn
        
    def callback(self):
        if self._callback:
            self._callback(self)
        
    def print_counter(self):
        if self.counter % 10 == 0:
            print(f"Counter at {self.counter}", time.time())
        
    def init_tissue(self, size=3):
        super().init_tissue(size=size)
        total_size = size**2 - 1
        # set north
        for i in range(size):
            self.G.nodes[i]["side"] = self.G.nodes[i].get("side", set()) | {"N"}
        
        # set south
        for i in range(size):
            self.G.nodes[total_size - i]["side"] = self.G.nodes[total_size - i].get("side", set()) | {"S"}
        
        # set west
        for i in range(size):
            self.G.nodes[i * size]["side"] = self.G.nodes[i * size].get("side", set()) | {"W"}
        
        # set east
        for i in range(size):
            self.G.nodes[total_size - i * size]["side"] = \
                self.G.nodes[total_size - i * size].get("side", set()) | {"E"}
            
        self.update_all_dist()
        
    def is_side_corner(self, node):
        return len(self.sides(node)) >= 2
            
    def diffuse_border(self, node):
        # if the border is a corner, we can just return
        # ideally, we shall never have a length of 3 ...
        
        if self.is_border_node(node):
            if len(self.G.nodes[node].get("side", set())) >= 2:
                return
            ngb1, ngb2 = self.get_border_neighbours(node)
            side1 = self.G.nodes[ngb1].get("side", set())
            side2 = self.G.nodes[ngb2].get("side", set())
            side = self.G.nodes[node].get("side", set())
            assert len(side1 & side2) == 1, f"side is not of the right size, " \
                f"should be one : {side1}, {side2}, {ngb1}, {ngb2}" \
                f" with node {node}, {side}"
            self.G.nodes[node]["side"] = side1 & side2
            
        else:
            if "side" in self.G.nodes[node]:
                del self.G.nodes[node]["side"]
                
    def diffuse_border_ngb(self, node):
        if not self.is_border_node(node):
            return
        
        for ngb in self.get_border_neighbours(node):
            if len(self.sides(ngb)) == 0:
                self.diffuse_border(ngb)
    
    def update_corner(self, node):
        if self.is_border_node(node):
            ngb1, ngb2 = self.get_border_neighbours(node)
            side1 = self.G.nodes[ngb1].get("side", set())
            side2 = self.G.nodes[ngb2].get("side", set())
            side_ref = self.G.nodes[node].get("side", set())
            
            if len(side1 & side2) == 0:
                self.G.nodes[ngb1]["side"] = side1 | side2
                if len(side1 | side2) == 3:
                    print("FAIL 1", side1, side2, ngb1, ngb2)
                    raise RuntimeError()
            
    # TODO override duplicate to set the border (NSEW)
    def duplicate(self, node, relax=False):
        self.counter += 1
        self.print_counter()
        new_node = super().duplicate(node)
        # switch corner in case new_node has taken the old node's place
        if not self.is_border_node(node) and self.is_side_corner(node):
            self.G.nodes[new_node]["side"] = self.sides(node)
            del self.G.nodes[node]["side"]
            self.diffuse_border(node)
        else:
            self.diffuse_border(new_node)
        self.callback()
        
        if relax:
            self.relax_step()
            
        return new_node
    
    # TODO override remove to check for corners
    def destroy(self, node, relax=False):
        self.counter += 1
        self.print_counter()
        self.update_corner(node)
        ngbs = list(self.ngb(node))
        super().destroy(node)
        for ngb in ngbs:
            self.diffuse_border_ngb(ngb)  # because of the remove + create trick in the destroy
        self.callback()
        
        if relax:
            self.relax_step()
        
    def relax_step(self):
        for i in range(2):
            self.relax()
            
        if self.get_main_metric() > 1e5:
            for i in range(2):
                self.relax()
    
    def edge_swapper(self):
        pass
    
    def update_all_dist(self):
        for direction in "NESW":
            self.update_dist(direction=direction)
            
    def quick_update_all_dist(self):
        print("changed", self.changed)
        for direction in "NESW":
            self.quick_update_dist(direction=direction)
        self.reset_cache_changed()
            
    def update_all_dist_for_action(self, action):
        for direction in "NESW":
            self.update_dist_for_action(action, direction=direction)
        
    def mean_NS(self):
        return np.mean([self.G.nodes[n]["dist_N"] + self.G.nodes[n]["dist_S"] for n in self.G.nodes])
    
    def mean_EW(self):
        return np.mean([self.G.nodes[n]["dist_E"] + self.G.nodes[n]["dist_W"] for n in self.G.nodes])
    
    def update_dist_for_action(self, action, direction="N"):
        pass
    
    def update_dist(self, direction="N"):
        """
        Update for each node its distance to each sides
        Direction is one of NSEW
        """
        key = "dist_" + direction
        queue = []
        nodes = set(self.G.nodes)
        # NORTH
        for i in self.G.nodes:
            n = self.G.nodes[i]
            if direction in n.get("side", set()):
                value = 0
            else:
                value = np.inf
                
            n[key] = value
            heappush(queue, (value, i))
            
        while queue and nodes:
            _, inode = heappop(queue)
            if inode not in nodes:
                continue
                
            nodes.remove(inode)
            node = self.G.nodes[inode]
            value = node[key] + 1
            for ngb in node["ngb"]:
                if value < self.G.nodes[ngb][key]:
                    self.G.nodes[ngb][key] = value
                    heappush(queue, (value, ngb))
                    
    def quick_update_dist(self, direction="N"):
        print("-"*80)
        print("direction", direction)
        key = "dist_" + direction
        queue = []
        to_collect = []
        nodes = self.changed.copy()
        
        for n in nodes:
            if n not in self.G.nodes:
                continue
            node = self.G.nodes[n]
            if direction in node.get("side", set()):
                value = 0
                heappush(queue, (value, n, False))
            else:
                value = np.inf
                to_collect.append(n)
                
            node[key] = value
            # value of dist, node n, whether it's a first trial (if 0 First is not necessary)
        print(queue)
        print(to_collect)
        
        for n in to_collect:
            val = self.collect_dist(n, key)
            if val != -1:
                print("Collect", np.inf, n)
                heappush(queue, (val, n, False))
            
        while queue and nodes:
            refval, inode, is_first = heappop(queue)
            print("Computing", refval, inode, is_first)
            if inode not in nodes:
                continue
                
            nodes.remove(inode)
            ls_updated, value = self.dispatch_dist(inode, key)
            for ngb in ls_updated:
                print("dispatch", value, ngb)
                heappush(queue, (value, ngb, False))
                    
    def dispatch_dist(self, inode, key):
        ls_ret = []
        node = self.G.nodes[inode]
        value = node[key] + 1
        for ngb in node["ngb"]:
            if value < self.G.nodes[ngb].get(key, np.inf):
                self.G.nodes[ngb][key] = value
                ls_ret.append(ngb)
        return ls_ret, value
    
    def collect_dist(self, inode, key):
        has_changed = False
        node = self.G.nodes[inode]
        value = node.get(key, np.inf)
        for ngb in node["ngb"]:
            other_val = self.G.nodes[ngb].get(key, np.inf) + 1
            print("COL", value, other_val)
            if value > other_val:
                self.G.nodes[inode][key] = other_val
                value = other_val
                has_changed = True
        print("has_changed", has_changed)
        return value if has_changed else -1
    
    def show_dist(self, k=5, iterations=1000, figsize=(10, 10), pos=None, hl_nodes=[],
                 print_dist=True):
        # green for NS, red for EW
        fig = plt.figure(figsize=figsize)
        if not pos:
            pos = nx.spring_layout(self.G, k=k, iterations=iterations)
        green = lambda x: format(max(0, 255 - (x["dist_N"] + x["dist_S"]) * 20), '02x')
        red = lambda x: format(max(0, 255 - (x["dist_E"] + x["dist_W"]) * 20), '02x')
        corner = lambda x: len(x.get("side", set())) >= 2
        
        def col_border(node):
            side = tuple(node.get("side", set()))
            return {
                ("N",): "#888822",  # yellow
                ("S",): "#228888",  # cyan
                ("E",): "#882288",  # purple
                ("W",): "#666666",  # grey
            }.get(side, "")
        
        def col_other(n):
            if n in hl_nodes:
                return "#FF4444"  # red
            return "#" + red(self.G.nodes[n]) + green(self.G.nodes[n]) + "ff"
            
        colors = ["#000000" if corner(self.G.nodes[n]) else 
                  col_border(self.G.nodes[n])
                  or col_other(n) for n in self.G.nodes]
        
        if print_dist:
            xi = 0.05
            delta = dict(N=np.array([-xi, 0]), S=np.array([xi, 0]),
                        E=np.array([0, -xi]), W=np.array([0, xi]))
            for d in "NESW":
                for n in self.G.nodes:
                    x, y = pos[n] + delta[d]
                    plt.text(x, y, self._dist(n, d))
                    
        
        nx.draw_networkx(self.G, pos, node_color=colors)
        return fig
                
    def _get_border_candidates(self, node):
        assert self.is_border_node(node)
        return [n for ngb in self.get_border_neighbours(node)
                for n in self.get_border_neighbours(ngb)
                if n != node]
    
    def _get_quick_candidates(self, node, dual_node):
        assert dual_node != -1, "Operation not allowed for dual -1"
        return [ngb for ngb in 
                self.get_intermediate_neighbours(dual_node, net="dual")
                if (ngb != node and (ngb, node) not in self.G.edges)]
    
    def _get_common_border_ngb(self, n_a, n_b):
        s_a = set(self.get_border_neighbours(n_a))
        s_b = set(self.get_border_neighbours(n_b))
        return list(s_a & s_b)[0]
    
    def get_func_action(self, name):
        if name == "build":
            return self.build_new_edge
        
        elif name == "replace":
            return self.replace_edge
        
        elif name == "setup":
            return self.setup_edge
    
    def build_new_edge_pattern(self, n, candidate, dual):
        pattern = [("+", (n, candidate))]
        return pattern
        
    @gpn_action
    def build_new_edge(self, n, candidate, dual):
        if (n, candidate) in self.G.edges:
            return
        
        if dual == -1:
            ref_node = self._get_common_border_ngb(n, candidate)
            if self.is_corner(ref_node):
                self.update_corner(ref_node)
            self._make_border_edge(ref_node, (n, candidate))
            self.diffuse_border(ref_node)
        else:
            self._make_edge(dual, (n, candidate))
            
        self.stabilize_around([n, candidate])
            
        return True
    
    def replace_edge_pattern(self, edge1, edge2):
        pattern = [("+", edge1), ("-", edge2)]
        return pattern
                
    @gpn_action
    def replace_edge(self, edge1, edge2):
        dual_node = self._remove_edge(edge1)
        # print("Dual node", dual_node)
        self._make_edge(dual_node, edge2)
        self.stabilize_around(list(set(edge1) | set(edge2)))
        
    def setup_edge_pattern(self, edge, dual_node, candidate):
        pattern = list()
        # setup the edge and delete the one chosen
        node1, node2 = edge
        if (node1, candidate) not in self.G.edges:
            pattern.append(("+", (node1, candidate)))
            
        if (node2, candidate) not in self.G.edges:
            pattern.append(("+", (node2, candidate)))
        
        # self._remove_edge(edge)  # WHY ???
        pattern.append(("-", edge))
        return pattern
    
    @gpn_action
    def setup_edge(self, edge, dual_node, candidate):
        # setup the edge and delete the one chosen
        node1, node2 = edge
        act1 = self.is_border_node(node1)
        act2 = self.is_border_node(node2)
        act_cand = self.is_border_node(candidate)
        if (node1, candidate) not in self.G.edges:
            other_dual = self._make_edge(dual_node, (node1, candidate))
            # update dual_node
            ingb = set(self.get_intermediate_neighbours(dual_node, net="dual"))
            if not ingb.issuperset({node2, candidate}):
                dual_node, other_dual = other_dual, dual_node
            
        if (node2, candidate) not in self.G.edges:
            self._make_edge(dual_node, (node2, candidate))
        
        self._remove_edge(edge)
        self.diffuse_border(candidate)
        self.stabilize_around(list(set(edge) | {candidate}))
        
    def relax(self):
        # prepare
        self.last_action = None
        if self.quick:
            self.quick_update_all_dist()
        else:
            self.update_all_dist()
        mean_NS = self.mean_NS()
        mean_EW = self.mean_EW()
        
        nodes = list(self.G.nodes)
        random.shuffle(nodes)
        
        # list all pain points
        candidates = list()
        for n in nodes:
            node = self.G.nodes[n]
            dist_NS = node["dist_N"] + node["dist_S"]
            dist_EW = node["dist_E"] + node["dist_W"]
            val_NS = dist_NS / mean_NS
            val_EW = dist_EW / mean_EW
            if dist_NS < 2:
                candidates.append(EnergyPoint(n, "NS", self.low_width_penalty(dist_NS)))
            elif val_NS >= 1.1 or val_NS <= 0.9:
                candidates.append(EnergyPoint(n, "NS", val_NS))
                
            if dist_EW < 2:
                candidates.append(EnergyPoint(n, "EW", self.low_width_penalty(dist_EW)))
            elif val_EW >= 1.1 or val_EW <= 0.9:
                candidates.append(EnergyPoint(n, "EW", val_EW))
                
        # lower is worse, so to be tested
                
        candidates.sort(reverse=False)
        
        # list all possible actions
        action_candidates = list()
        for pt in candidates[:4]:
            tot_action = 0
            for action in self.enumerate_actions(pt):
                tot_action += 1
                if tot_action > 100:
                    print(pt.node, self.ngb(pt.node))
                    print(locals())
                    raise RuntimeError()
                self.estimate_action(action, mean_NS, mean_EW)
                action_candidates.append(action)
                
        # higher score is better to do !!
                
        action_candidates.sort(reverse=True)
        
        for action in action_candidates[:2]:
            self.try_action(action)
            
        self.keep_best_trial()
        
    def enumerate_actions(self, pt):
        if pt.to_shorten():
            for x in self._shorten(pt):
                yield x
        else:
            for x in self._maxen(pt):
                yield x
                
    def _quick_shorten(self, pt):
        ingbs = self.get_intermediate_neighbours(pt.node)
        
        # gather all candidates without any filter
        candidates = []
        for ingb in ingbs:
            if ingb == -1:
                ls_c = self._get_border_candidates(pt.node)
            else:
                ls_c = self._get_quick_candidates(pt.node, ingb)
            for c in ls_c:
                args = (pt.node, c, ingb)
                action = RelaxAction(name="build", args=args, 
                                     pattern=self.build_new_edge_pattern(*args))
                yield action
                
    def _shorten(self, pt):
        # maybe it could be simplified later
        for x in self._quick_shorten(pt):
            yield x
            
        ingbs = set(self.get_intermediate_neighbours(pt.node)) - {-1}
        
        for ingb in ingbs:
            dngbs = self.ngb(ingb, net="dual")
            
            for edge_dngb in dngbs:
                node_dngb = self.get_other_node(edge_dngb, ingb)
                if node_dngb in (set(ingbs) | {-1}):
                    continue
                    
                # if #ngb == 2
                p_edge = self.dual(edge_dngb)
                if len(self.ngb(p_edge[0])) == 2 or len(self.ngb(p_edge[1])) == 2:
                    continue
                    
                pngbs = self.get_intermediate_neighbours(node_dngb, net="dual")
                
                for pngb in pngbs:
                    # no edge
                    if (pt.node, pngb) in self.G.edges:
                        continue
                        
                    # HERE creating action
                    args = (self.dual(edge_dngb), (pt.node, pngb))
                    action = RelaxAction(name="replace", args=args, 
                                         pattern=self.replace_edge_pattern(*args))
                    
                    yield action
    
    def _maxen(self, pt):
        if len(self.ngb(pt.node)) == 2:  # this is a bit chiant
            return
        
        for ngb in self.ngb(pt.node):
            if len(self.ngb(ngb)) == 2:
                continue
                
            dual_nodes = set(self.dual((pt.node, ngb))[:2]) - {-1}
            for dual_node in dual_nodes:
                pngbs = set(self.get_intermediate_neighbours(dual_node, net="dual")) - {pt.node, ngb}
                for pngb in pngbs:
                    # don't allow triple border nodes
                    if self.is_border_node(pt.node) \
                        and self.is_border_node(ngb) \
                        and self.is_border_node(pngb):
                        continue
                    args = ((pt.node, ngb), dual_node, pngb)
                    action = RelaxAction(name="setup", args=args,
                                        pattern=self.setup_edge_pattern(*args))

                    yield action
                        
    def estimate_action(self, action, mean_NS, mean_EW):
        old_score_NS = []
        new_score_NS = []
        old_score_EW = []
        new_score_EW = []
        for node in action.get_nodes():
            ngbs = set(self.ngb(node)) | action.more(node) - action.less(node)
            old_score_NS.append(self._dist(node, "NS"))
            old_score_EW.append(self._dist(node, "EW"))
            dist = dict(N=1e9, E=1e9, S=1e9, W=1e9)
            for ngb in ngbs:
                for direction in "NESW":
                    dist[direction] = min(dist[direction], self._dist(ngb, direction) + 1)
            new_score_NS.append(dist["N"] + dist["S"])
            new_score_EW.append(dist["W"] + dist["E"]) # are estimates good ?
                
        
        score = np.sum((np.array(old_score_NS) - mean_NS)**4)
        score += np.sum((np.array(old_score_EW) - mean_EW)**4)
        score += np.sum([self.low_width_penalty(x) for x in (old_score_NS + old_score_EW)])
        score -= np.sum((np.array(new_score_NS) - mean_NS)**4)
        score -= np.sum((np.array(new_score_EW) - mean_EW)**4)
        score -= np.sum([self.low_width_penalty(x) for x in (new_score_NS + new_score_EW)])
        action.score = score
    
    def try_action(self, action):
        try:
            gpn = self.copy()
            gpn.run_action(action)
            
        except:
            pass
            # print(f"Action {action} failed")
            
        else:
            self.trials.append(gpn)
    
    def run_action(self, action):
        func = self.get_func_action(action.name)
        self.last_action = action
        func(*action.args)
        
    def keep_best_trial(self):
        # a bit ugly but should work
        best = min(self.trials + [self], key=lambda x: x.get_main_metric(update=True))
        self.trials.clear()
        # print(f"Action ran : {best.last_action}")
        self.__dict__ = best.__dict__
        self.callback()
        
    def low_width_penalty(self, dist):
        if dist < 1:
            return 1e6
        elif dist < 2:
            return 1e5
        else:
            return 0
        
    def low_width_penalty_old(self, dist):
        if dist < 1:
            return 1e6
        elif dist < 2:
            return 1e5
        elif dist < 3:
            return 1e4
        elif dist < 4:
            return 1e3
        else:
            return 0
    
    def get_all_dists(self):
        l_NS, l_EW, l_both = list(), list(), list()
        nodes = list(self.G.nodes)
        candidates = list()
        for n in nodes:
            node = self.G.nodes[n]
            val_NS = (node["dist_N"] + node["dist_S"])
            val_EW = (node["dist_E"] + node["dist_W"])
            l_NS.append(val_NS)
            l_EW.append(val_EW)
            l_both.append(val_NS + val_EW)
            
        return l_NS, l_EW, l_both
    
    def get_main_metric(self, update=False):
        if update:
            if self.quick:
                self.quick_update_all_dist()
            else:
                self.update_all_dist()
                
        lA, lB, lC = self.get_all_dists()
        # xA, xB, xC = np.std(lA) / np.mean(lA), np.std(lB) / np.mean(lB), np.std(lC) / np.mean(lC)
        xA, xB, xC = std4(lA), std4(lB), std4(lC)
        correctorA = np.sum([self.low_width_penalty(x) for x in lA])
        correctorB = np.sum([self.low_width_penalty(x) for x in lB])
        # correctorA = np.sum(np.array(lA) < 4) * 1e3 + np.sum(np.array(lA) < 3) * 1e5
        # correctorB = np.sum(np.array(lB) < 4) * 1e3 + np.sum(np.array(lB) < 3) * 1e5
        return xA + xB + xC + correctorA + correctorB
    
    def print_dist_metrics(self):
        lA, lB, lC = self.get_all_dists()
        xA, xB, xC = np.std(lA) / np.mean(lA), np.std(lB) / np.mean(lB), np.std(lC) / np.mean(lC)
        main = self.get_main_metric()
        print(f"Score are for NS : {xA}, EW : {xB}, both : {xC}, main : {main}")
        return xA + xB + xC
    
    # TODO, find a better way
    def _dist(self, n, axis):
        node = self.G.nodes[n]
        return sum([node["dist_" + str(x)] for x in axis])