import pytest
from lib.sde.util.formula import And, Or, Not, Var
from lib.sde.mutate import tree_add_gene, tree_remove_gene, tree_substitute_gene


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


def test_tree_add_gene(tree):
    labels = set(range(1, 5))
    for i in range(10):
        new_tree = tree_add_gene(tree, labels)
        # there is a NOT NOT problem
        assert len(new_tree.op_list()) + len(new_tree.labels_set()) > 8


def test_tree_substitute_gene(tree):
    labels = set(range(1, 5))
    for i in range(10):
        new_tree = tree_substitute_gene(tree, labels)
        assert tree.labels_set() != new_tree.labels_set()


def test_tree_remove_gene(tree):
    labels = set(range(1, 5))
    for i in range(10):
        new_tree = tree_remove_gene(tree, labels)
        assert len(tree.labels_set()) > len(new_tree.labels_set())
