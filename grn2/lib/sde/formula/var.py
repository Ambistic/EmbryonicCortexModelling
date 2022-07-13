from lib.sde.formula import Formula


class Var(Formula):
    """
    Var can handle both dict and list-like values
    """
    nb_children = 0

    def __init__(self, name, sign: bool = True, weight: float = 1.0):
        super().__init__()
        self.name = name
        self.children = []
        self.sign = sign  # True for positive
        self.weight = weight

    def is_positive(self):
        return self.sign

    def __call__(self, values):
        if self.is_positive():
            return values[self.name]
        else:
            return 1 - values[self.name]

    def __repr__(self):
        prefix = "-" if self.is_positive() else ""
        return f"{prefix}{self.weight:.2f}x{self.name}"

    def _to_json(self):
        return self.name

    def to_dict(self):
        return dict(op="Var", val=self.name, sign=self.sign, weight=self.weight)

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
        if self.is_positive():
            return lambda x: x[self.name]
        else:
            return lambda x: 1 - x[self.name]


class Zero(Formula):
    nb_children = 0
    sign = True

    def __init__(self, name=None, weight=1.0, sign=True):
        super().__init__()
        self.name = "0z"
        self.weight = weight

    @staticmethod
    def is_positive():
        return True

    def __call__(self, values):
        return 0

    def __repr__(self):
        return f"{self.weight:.2f}x0z"

    def _to_json(self):
        return "0z"

    def to_dict(self):
        return dict(op="Zero", weight=self.weight)

    def labels_set(self):
        return {"0z"}

    def as_lambda(self):
        return lambda x: 0

    def replace(self, old, new):
        raise NotImplementedError("Cannot replace a Zero")


class One(Formula):
    nb_children = 0
    sign = True

    def __init__(self, name=None, weight=1.0, sign=True):
        super().__init__()
        self.name = "1z"
        self.weight = weight

    @staticmethod
    def is_positive():
        return True

    def __call__(self, values):
        return 1

    def __repr__(self):
        return f"{self.weight:.2f}x1z"

    def _to_json(self):
        return "1z"

    def to_dict(self):
        return dict(op="One", weight=self.weight)

    def labels_set(self):
        return {"1z"}

    def as_lambda(self):
        return lambda x: 1

    def replace(self, old, new):
        raise NotImplementedError("Cannot replace a One")


def build_var(name, sign, weight):
    if name == "1z":
        return One(weight=weight)
    elif name == "0z":
        return Zero(weight=weight)
    else:
        return Var(name=name, sign=sign, weight=weight)
