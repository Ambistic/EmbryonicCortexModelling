import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import operator
"""
This file implements short functions and helper classes



"""

#############
# FUNCTIONS #
#############

def add(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1])

def sub(t1, t2):
    return (t1[0] - t2[0], t1[1] - t2[1])

def comp(a: tuple, b: tuple) -> bool:
    if a == b:
        return True
    if a[0] == b[1] and a[1] == b[0] and a[2:] == b[2:]:
        return True
    return False

def unique(ls):  # onl for hashable stuff
    return list(set(ls))

def index_of(x, ls):
    if type(x) != tuple:
        return ls.index(x)
    
    for i in range(len(ls)):
        if comp(x, ls[i]):
            return i
    raise ValueError(f"{x} is not found in {ls}")
    
def jaccard_coefficient(G, n1, n2):
    return list(nx.jaccard_coefficient(G, [(n1, n2)]))[0][2]

flatten = lambda t: [item for sublist in t for item in sublist]

def cycle_from_pairs(ls_pairs):  # shall be called ingb_from_pairs
    cycle = CircularList()
    G = nx.Graph()
    for e in ls_pairs:
        G.add_edge(*e)
    old = list(G.nodes)[0]
    n = list(G.neighbors(old))[0]
    while not G.nodes[n].get("done", False):
        G.nodes[n]["done"] = True
        cycle.append(n)
        next_ = set(G.neighbors(n)).difference({old})
        if len(next_) != 1:
            raise ValueError("ls_pairs is not correct " + str(ls_pairs))
        next_ = list(next_)[0]
        old, n = n, next_
        
    return cycle

def cycle_two_pairs(ls_p):
    ls = list(map(list, ls_p))
    if ls[0][0] == ls[1][0]:
        ls[0][0], ls[0][1] = ls[0][1], ls[0][0]
    
    return list(map(tuple, ls))

def cycle_from_ordered_list_pairs(ls_p):
    """
    The case where len(ls_p) == 2 is not handled
    How to know which one is the first ?
    Let's say it doesn't matter
    """
    if len(ls_p) == 2:
        return cycle_two_pairs(ls_p)
    
    ls = list(map(list, ls_p))
    if ls[0][0] in ls[1][:2]:
        ls[0][0], ls[0][1] = ls[0][1], ls[0][0]
        
    for i in range(1, len(ls)):
        if ls[i][1] in ls[i - 1]:
            ls[i][0], ls[i][1] = ls[i][1], ls[i][0]
            
    return list(map(tuple, ls))

def list_from_cycle_dual(dict_):
    ls = flatten(list(dict_.values()))
    return list(set(ls))  # for unique
    
#############
#  CLASSES  #
#############

class Ltuple(tuple):
    """
    ltuple implements a loose tuple class
    where index 0 and 1 are invertible
    """
    def __eq__(self, other):
        return comp(tuple(self), tuple(other))
    
    def __hash__(self):
        if self[0] < self[1]:
            return hash(tuple(self))
        
        x = list(self)
        x[0], x[1] = x[1], x[0]
        return hash(tuple(x))
    

class CircularList(list):
    def __getitem__(self, x):
        if isinstance(x, slice):
            return CircularList([self[x] for x in self._rangeify(x)])

        index = operator.index(x)
        try:
            return super().__getitem__(index % len(self))
        except ZeroDivisionError:
            raise IndexError('list index out of range')

    def _rangeify(self, slice):
        start, stop, step = slice.start, slice.stop, slice.step
        if start is None:
            start = 0
        if stop is None:
            stop = len(self)
        if step is None:
            step = 1
        return range(start, stop, step)
    
    def next(self, value):
        return self[self.index(value) + 1]
    
    def match_pattern(self, pattern: list):
        if len(pattern) == 0:
            return True
        
        current = pattern[0]
        idx = self.index(current)
        
        for i in range(len(pattern)):
            if not comp(pattern[i], current):
                return False
            current = self.next(current)
            
        return True
    
    def replace(self, old, new):
        for i in range(len(self)):
            if super().__getitem__(i) == old:
                super().__setitem__(i, new)
                break

