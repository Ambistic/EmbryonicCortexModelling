import random

import numpy as np

from lib.sde.formula import Var, And, Or, One, Zero, build_var


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


def generate_random_tree(regulator_labels: set, max_number=3, prob_gene=0.8, mode="and"):
    if mode == "and":
        cls_ = And
        bonus = {"0z"}
    elif mode == "or":
        cls_ = Or
        bonus = {"1z"}
    else:
        raise ValueError(f"Unknown mode {mode}")

    regulator_labels = set(regulator_labels) | bonus
    ls = []
    while random.random() < prob_gene and max_number > 0 and len(regulator_labels):
        name = random.choice(tuple(regulator_labels))
        regulator_labels.remove(name)
        ls.append(build_var(name, sign=random.choice([True, False]), weight=random.random()))
        max_number -= 1

    return cls_(*ls)
