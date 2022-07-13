#!/usr/bin/env python
# coding: utf-8
from abc import ABC, abstractmethod
from functools import reduce
import random
from jax import jit

from lib.sde.formula.helper import formula_from_dict


class Formula(ABC):
    parent = None
    children = []

    def __init__(self):
        self._cache = dict()

    def is_root(self):
        return self.parent is None

    def set_parent(self, parent):
        self.parent = parent
    
    @abstractmethod
    def __call__(self, values):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass
    
    def print(self):
        print(repr(self))

    def to_json(self):
        return self._to_json()

    @abstractmethod
    def _to_json(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    def labels_set(self):
        return reduce(set.union, [child.labels_set() for child in self.children], set())

    def op_list(self):
        return reduce(lambda x, y: x + y, [child.op_list() for child in self.children], []) + [self]

    def gene_list(self):
        return reduce(lambda x, y: x + y, [child.gene_list() for child in self.children], [])

    def random_op(self):
        return random.choice(self.op_list())

    def random_gene(self):
        return random.choice(self.gene_list())

    @abstractmethod
    def replace(self, old, new):
        pass

    def update(self):
        self._cache = dict()
        for child in self.children:
            child.update()

    def copy(self):
        return formula_from_dict(self.to_dict())

    def get_compiled(self):
        if "compiled" not in self._cache:
            self._cache["compiled"] = jit(self.as_lambda())
        return self._cache["compiled"]

    @abstractmethod
    def as_lambda(self):
        pass
