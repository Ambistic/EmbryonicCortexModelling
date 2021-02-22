def comp(a: tuple, b: tuple) -> bool:
    if a == b:
        return True
    if a[0] == b[1] and a[1] == b[0] and a[2:] == b[2:]:
        return True
    return False

def unique(ls):
    return list(set(ls))

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

def index_of(x, ls):
    if type(x) != tuple:
        return ls.index(x)
    
    for i in range(len(ls)):
        if comp(x, ls[i]):
            return i
    raise ValueError(f"{x} is not found in {ls}")
