from lib.sde.formula import Formula


class And(Formula):
    def __init__(self, *children):
        super().__init__()
        self.nb_children = len(children)
        self.children = list(children)
        for child in self.children:
            child.set_parent(self)

    def __call__(self, values):
        val = 1
        for child in self.children:
            val = val * (1 - child.weight * (1 - child(values)))
        return val

    def __repr__(self):
        return "(" + " AND ".join(map(repr, self.children)) + ")"

    def _to_json(self):
        raise NotImplementedError()

    def to_dict(self):
        return dict(op='And', children=[child.to_dict() for child in self.children])

    def replace(self, old, new):
        idx = self.children.index(old)
        self.children[idx] = new
        new.set_parent(self)

    def as_lambda(self):
        def func(x):
            val = 1
            for child in self.children:
                val = val * (1 - child.weight * (1 - child.as_lambda()(x)))
            return val
        return func

    def add_child(self, child):
        child.set_parent(self)
        self.children.append(child)
        self.nb_children += 1

    def remove_child(self, child):
        self.children.remove(child)
        self.nb_children -= 1


class Or(Formula):
    def __init__(self, *children):
        super().__init__()
        self.nb_children = len(children)
        self.children = list(children)
        for child in self.children:
            child.set_parent(self)

    def __call__(self, values):
        val = 0
        for child in self.children:
            x = child(values)
            val = val + x * child.weight - val * x * child.weight
        return val

    def __repr__(self):
        return "(" + " OR ".join(map(repr, self.children)) + ")"

    def _to_json(self):
        raise NotImplementedError()

    def to_dict(self):
        return dict(op='Or', children=[child.to_dict() for child in self.children])

    def replace(self, old, new):
        idx = self.children.index(old)
        self.children[idx] = new
        new.set_parent(self)

    def as_lambda(self):
        def func(x):
            val = 0
            for child in self.children:
                y = child.as_lambda()(x)
                val = val + x * child.weight - val * y * child.weight
            return val

        return func

    def add_child(self, child):
        child.set_parent(self)
        self.children.append(child)
        self.nb_children += 1

    def remove_child(self, child):
        self.children.remove(child)
        self.nb_children -= 1
