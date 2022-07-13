import random

import numpy as np

from lib.sde.util.formula import Var, And, Or, Not, SignedVar


_list_operators = [And, Or, Not]
_list_operators_2 = [And, Or]


def generate_random_init():
    return np.random.uniform(0, 3)


def generate_random_noise():
    return np.random.uniform(0, 10)


def generate_random_expr():
    return np.random.uniform(0, 10)


def generate_random_deg():
    return np.random.uniform(0, 10)


def generate_random_thr():
    return np.random.uniform(0, 10)


def generate_random_theta():
    return np.random.uniform(0, 10)


def generate_random_m():
    return np.random.uniform(1, 10)


def generate_random_b():
    return np.random.uniform(1, 10)


def generate_random_expr2():
    return np.random.uniform(0, 5)


def generate_random_deg2():
    return np.random.uniform(0, 3)


def generate_random_asym():
    return np.random.uniform(0.5, 10)


class NotEnoughGeneException(Exception):
    pass


def pick_and_remove_gene(gene_labels: set):
    if len(gene_labels) == 0:
        raise NotEnoughGeneException()

    selected_gene = random.choice(tuple(gene_labels))
    gene_labels.remove(selected_gene)
    return Var(selected_gene), gene_labels


def generate_random_tree(gene_labels: set, max_depth=2, prob_gene=0.5):
    if not isinstance(gene_labels, set):
        gene_labels = set(gene_labels)

    if np.random.random() < prob_gene or max_depth == 0:
        # Gene
        tree, _ = pick_and_remove_gene(gene_labels)
        return tree

    else:
        # Tree
        selected_op = random.choice(_list_operators)
        nb_subtrees = selected_op.nb_children

        try:
            subtrees = []
            current_gene_labels = gene_labels.copy()
            for _ in range(nb_subtrees):
                subtree, current_gene_labels = _generate_random_subtree(
                    gene_labels=current_gene_labels,
                    max_depth=max_depth - 1,
                    prob_gene=prob_gene
                )
                subtrees.append(subtree)

        except NotEnoughGeneException:
            tree, _ = pick_and_remove_gene(gene_labels)
            return tree

        return selected_op(*subtrees)


def _generate_random_subtree(gene_labels: set, max_depth=2, prob_gene=0.5):
    gene_labels = gene_labels.copy()
    if np.random.random() < prob_gene or max_depth == 0:
        # Gene
        return pick_and_remove_gene(gene_labels)

    else:
        # Tree
        selected_op = random.choice(_list_operators)
        nb_subtrees = selected_op.nb_children

        try:
            subtrees = []
            for _ in range(nb_subtrees):
                subtree, gene_labels = _generate_random_subtree(
                    gene_labels=gene_labels,
                    max_depth=max_depth - 1,
                    prob_gene=prob_gene
                )
                subtrees.append(subtree)

        except NotEnoughGeneException:
            return pick_and_remove_gene(gene_labels)

        return selected_op(*subtrees), gene_labels


def pick_and_remove_gene_2(gene_labels: set):
    if len(gene_labels) == 0:
        raise NotEnoughGeneException()

    selected_gene = random.choice(tuple(gene_labels))
    gene_labels.remove(selected_gene)
    sign = bool(random.randint(0, 1))
    return SignedVar(selected_gene, sign), gene_labels


def generate_random_tree_2(gene_labels: set, max_depth=2, prob_gene=0.5):
    if not isinstance(gene_labels, set):
        gene_labels = set(gene_labels)
    try:
        res, _ = _generate_random_subtree_2(gene_labels, max_depth=max_depth, prob_gene=prob_gene)
    except NotEnoughGeneException:
        tree, _ = pick_and_remove_gene_2(gene_labels)
        return tree
    return res


def _generate_random_subtree_2(gene_labels: set, max_depth=2, prob_gene=0.5):
    gene_labels = gene_labels.copy()
    if np.random.random() < prob_gene or max_depth == 0:
        # Gene
        return pick_and_remove_gene_2(gene_labels)

    else:
        # Tree
        selected_op = random.choice(_list_operators_2)
        nb_subtrees = selected_op.nb_children

        try:
            subtrees = []
            for _ in range(nb_subtrees):
                subtree, gene_labels = _generate_random_subtree_2(
                    gene_labels=gene_labels,
                    max_depth=max_depth - 1,
                    prob_gene=prob_gene
                )
                subtrees.append(subtree)

        except NotEnoughGeneException:
            return pick_and_remove_gene_2(gene_labels)

        return selected_op(*subtrees), gene_labels
