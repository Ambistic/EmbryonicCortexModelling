import json
import numpy as np
try:
    import jax.numpy as jnp
except ImportError:
    jnp = np


def batch(sequence, n=1):
    lg = len(sequence)
    for ndx in range(0, lg, n):
        yield sequence[ndx:ndx + n]


def finebatch(seq, base, mul):
    ln = len(seq)
    p_mul = int(np.log(ln / base) / np.log(mul))
    ndx = 0
    while ndx < ln:
        while p_mul > 0 and (ln - ndx) < base * mul**p_mul:
            p_mul -= 1
        size = base * mul**p_mul
        yield seq[ndx:ndx + size]
        ndx += size


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, jnp.ndarray):
            obj = np.array(obj)

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        return super(NpEncoder, self).default(obj)
