#!/usr/bin/env python
# coding: utf-8
from abc import ABC, abstractmethod
from functools import reduce
import random
from jax import jit


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
    
    def __and__(self, other):
        return And(self, other)
    
    def __or__(self, other):
        return Or(self, other)
    
    def __neg__(self):
        return Not(self)
    
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
        return reduce(set.union, [child.labels_set() for child in self.children])

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


class And(Formula):
    nb_children = 2

    def __init__(self, F1: Formula, F2: Formula):
        super().__init__()
        self.F1 = F1
        self.F2 = F2
        self.children = [F1, F2]
        F1.set_parent(self)
        F2.set_parent(self)
    
    def __call__(self, values):
        return self.F1(values) * self.F2(values)
    
    def __repr__(self):
        return "(" + repr(self.F1) + " AND " + repr(self.F2) + ")"

    def _to_json(self):
        return "{'op': 'And', F1: {F1}, F2: {F2}}".format(F1=self.F1, F2=self.F2)

    def to_dict(self):
        return dict(op='And', F1=self.F1.to_dict(), F2=self.F2.to_dict())

    def replace(self, old, new):
        if self.F1 is old:
            self.F1 = new
            self.children[0] = new
            new.set_parent(self)

        elif self.F2 is old:
            self.F2 = new
            self.children[1] = new
            new.set_parent(self)

        else:
            raise ValueError(f"{old} is not found in {self}")

    def other_child(self, child):
        if self.F1 is child:
            return self.F2

        elif self.F2 is child:
            return self.F1

    def as_lambda(self):
        """
        The x param in the lambda function is the vector transmitted
        """
        return lambda x: self.F1.as_lambda()(x) * self.F2.as_lambda()(x)


class Or(Formula):
    nb_children = 2

    def __init__(self, F1, F2):
        super().__init__()
        self.F1 = F1
        self.F2 = F2
        self.children = [F1, F2]
        F1.set_parent(self)
        F2.set_parent(self)
    
    def __call__(self, values):
        f1, f2 = self.F1(values), self.F2(values)
        return f1 + f2 - f1 * f2
    
    def __repr__(self):
        return "(" + repr(self.F1) + " OR " + repr(self.F2) + ")"

    def _to_json(self):
        return "{'op': 'Or', F1: {F1}, F2: {F2}}".format(F1=self.F1, F2=self.F2)

    def to_dict(self):
        return dict(op='Or', F1=self.F1.to_dict(), F2=self.F2.to_dict())

    def replace(self, old, new):
        if self.F1 is old:
            self.F1 = new
            self.children[0] = new
            new.set_parent(self)

        elif self.F2 is old:
            self.F2 = new
            self.children[1] = new
            new.set_parent(self)

        else:
            raise ValueError(f"{old} is not found in {self}")

    def other_child(self, child):
        if self.F1 is child:
            return self.F2

        elif self.F2 is child:
            return self.F1

        else:
            raise ValueError(f"{child} is not found in {self}")

    def as_lambda(self):
        """
        The x param in the lambda function is the vector transmitted
        """
        return lambda x: 1 + (1 - self.F1.as_lambda()(x)) * (self.F2.as_lambda()(x) - 1)


class Not(Formula):
    nb_children = 1

    def __init__(self, F):
        super().__init__()
        self.F = F
        self.children = [F]
        F.set_parent(self)
    
    def __call__(self, values):
        return 1 - self.F(values)
    
    def __repr__(self):
        return "NOT " + repr(self.F)

    def _to_json(self):
        return "{'op': 'Not', F: {F}}".format(F=self.F)

    def to_dict(self):
        return dict(op='Not', F=self.F.to_dict())

    def replace(self, old, new):
        if self.F is old:
            self.F = new
            self.children[0] = new
            new.set_parent(self)

        else:
            raise ValueError(f"{old} is not found in {self}")

    def as_lambda(self):
        """
        The x param in the lambda function is the vector transmitted
        """
        return lambda x: 1 - self.F.as_lambda()(x)


class Var(Formula):
    """
    Var can handle both dict and list-like values
    """
    nb_children = 0

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.children = []
    
    def __call__(self, values):
        return values[self.name]
    
    def __repr__(self):
        return str(self.name)

    def _to_json(self):
        return self.name

    def to_dict(self):
        return dict(op="Var", val=self.name)

    def labels_set(self):
        return {self.name}

    def op_set(self):
        return []

    def gene_list(self):
        return [self]

    def replace(self, old, new):
        raise NotImplementedError("Cannot replace a Var")

    def as_lambda(self):
        """
        The x param in the lambda function is the vector transmitted
        """
        return lambda x: x[self.name]


class SignedVar(Formula):
    """
    Var can handle both dict and list-like values
    """
    nb_children = 0

    def __init__(self, name, sign: bool):
        super().__init__()
        self.name = name
        self.children = []
        self.sign = sign  # True for positive

    def is_positive(self):
        return self.sign

    def __call__(self, values):
        if self.is_positive():
            return values[self.name]
        else:
            return 1 - values[self.name]

    def __repr__(self):
        prefix = "-" if self.is_positive() else ""
        return f"{prefix}{self.name}"

    def _to_json(self):
        return self.name

    def to_dict(self):
        return dict(op="SignedVar", val=self.name, sign=self.sign)

    def labels_set(self):
        return {self.name}

    def op_set(self):
        return []

    def gene_list(self):
        return [self]

    def replace(self, old, new):
        raise NotImplementedError("Cannot replace a WeightedVar")

    def as_lambda(self):
        """
        The x param in the lambda function is the vector transmitted
        """
        if self.is_positive():
            return lambda x: x[self.name]
        else:
            return lambda x: 1 - x[self.name]


def formula_from_dict(dictf):
    op = dictf["op"]

    if op == "And":
        return And(formula_from_dict(dictf["F1"]),
                   formula_from_dict(dictf["F2"]))

    if op == "Or":
        return Or(formula_from_dict(dictf["F1"]),
                  formula_from_dict(dictf["F2"]))

    if op == "Not":
        return Not(formula_from_dict(dictf["F"]))

    if op == "Var":
        return Var(dictf["val"])

    if op == "SignedVar":
        return SignedVar(dictf["val"], dictf["sign"])


Ops = [And, Or, Not]
