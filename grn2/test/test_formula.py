import pytest
import numpy as np
import time
from lib.sde.formula.formula import Not, Formula, Ops
from lib.sde.formula.operator import And, Or
from lib.sde.formula import Var
import jax.numpy as jnp


@pytest.fixture
def tree():
    my_tree = And(
        Or(
            Var(1),
            Var(2)
        ),
        Not(
            Var(4)
        )
    )
    return my_tree


def test_labels_set(tree: Formula):
    assert tree.labels_set() == {1, 2, 4}


def test_op_set(tree: Formula):
    assert all([isinstance(x, tuple(Ops)) for x in list(tree.op_set())])
    assert len(tree.op_set()) == 3


def test_lambda(tree: Formula):
    x = np.array([0.3, 0.4, 0.2, 0.1, 0.5])
    reference = 0.26
    assert tree(x) == reference
    assert tree.as_lambda()(x) == reference
    assert tree.get_compiled()(x) == reference


def test_perf_lambda(tree: Formula):
    print()
    x = np.array([0.3, 0.4, 0.2, 0.1, 0.5])

    start_time = time.time()
    [tree(x) for _ in range(1000)]
    print("--- Basic  : %s seconds ---" % (time.time() - start_time))

    lam = tree.as_lambda()
    start_time = time.time()
    [lam(x) for _ in range(1000)]
    print("--- Lambda : %s seconds ---" % (time.time() - start_time))

    comp = tree.get_compiled()
    start_time = time.time()
    [comp(x) for _ in range(1000)]
    print("--- Jit    : %s seconds ---" % (time.time() - start_time))

    comp = tree.get_compiled()
    x = jnp.array([0.3, 0.4, 0.2, 0.1, 0.5])
    start_time = time.time()
    [comp(x) for _ in range(1000)]
    print("--- JaxJit : %s seconds ---" % (time.time() - start_time))

