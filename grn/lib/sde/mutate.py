from lib.sde.gene.gene import *
from lib.sde.grn.grn import GRNMain
from lib.sde.grn.grn2 import GRNMain2
from lib.sde.grn.grn3 import GRNMain3
from lib.sde.grn.grn4 import GRNMain4
from lib.sde.gene.gene4 import GeneMain4
from lib.sde.util.formula import And, Or, Not, Var, Formula
import random

from lib.sde.util.initializer import generate_random_init, generate_random_noise, generate_random_expr, \
    generate_random_deg, generate_random_thr, generate_random_theta, generate_random_m, generate_random_b, \
    generate_random_tree, generate_random_expr2, generate_random_deg2, generate_random_asym

_op = [And, Or, Not]


def mutate_grn(grn: GRNMain):
    one_gene = random.choice(grn.genes)
    mutate_gene(one_gene)


def _mutate_grn2(grn: GRNMain2):
    one_gene = random.choice(grn.genes)
    mutate_gene2(one_gene)
    
    
def _mutate_grn3(grn: GRNMain2):
    one_gene = random.choice(grn.genes)
    mutate_gene3(one_gene)


def _mutate_grn4(grn: GRNMain4):
    one_gene = random.choice(grn.genes)
    mutate_gene4(one_gene)


def mutate_grn2(grn: GRNMain2):
    grn.set_mutable()
    _mutate_grn2(grn)
    grn.compile()
    
    
def mutate_grn3(grn: GRNMain3):
    grn.set_mutable()
    _mutate_grn3(grn)
    grn.compile()


def mutate_var(grn: GRNMain4):
    if "nb_mut" in grn.var:
        r = random.random()
        if r < 0.01:
            grn.var["nb_mut"] += 1
        elif r < 0.02:
            grn.var["nb_mut"] = max(1, grn.var["nb_mut"] - 1)


def mutate_grn4(grn: GRNMain4):
    nb_mut = grn.var.get("nb_mut", 1)
    mutate_var(grn)
    grn.set_mutable()
    for i in range(nb_mut):
        _mutate_grn4(grn)
    grn.compile()


def multi_mutate_grn2(grn: GRNMain2, value=1, method="poisson"):
    """
    Mutate a grn multiple times. The function cannot mutate 0 time.
    :param grn: The grn to be mutated (in place)
    :param value: The value, behaviour depends on the method
    :param method: one of "poisson" or "fixed", if "poisson" then the number
    of mutation is random from poisson distribution. If "fixed" then mutate the exact
    `value` times
    """
    if method == "fixed":
        number = value
    elif method == "poisson":
        number = max(1, np.random.poisson(value))
    else:
        raise ValueError(f"Not handled {method} for method of mutation number")

    grn.set_mutable()
    for i in range(number):
        _mutate_grn2(grn)
    grn.compile()
    
    
def mutate_gene2(gene: GeneMain, param_prob=0.5):
    if random.random() < param_prob:
        mutate_param2(gene)

    else:
        gene.tree = mutate_tree(gene.tree, gene.get_labels_not_in_tree())  # TODO add the labels here !!!
        
        
def mutate_gene3(gene: GeneMain, param_prob=0.3, smooth_param_prob=0.3):
    r = random.random()
    if r < param_prob:
        mutate_param3(gene)
        
    elif r < (param_prob + smooth_param_prob):
        mutate_smooth_param3(gene)

    else:
        gene.tree = mutate_tree(gene.tree, gene.get_labels_not_in_tree())  # TODO add the labels here !!!


def mutate_gene4(gene: GeneMain):
    r = random.random()
    param_prob = 0.05
    smooth_param_prob = 0.65

    if r < param_prob:
        mutate_param3(gene)

    elif r < (param_prob + smooth_param_prob):
        mutate_smooth_param3(gene)

    else:
        gene.tree = mutate_tree(gene.tree, gene.get_labels_not_in_tree())  # TODO add the labels here !!!


def mutate_gene5(gene: GeneMain4, generate_funcs: dict = None):
    r = random.random()
    reinit_prob = 0.05
    smooth_param_prob = 0.65

    if r < reinit_prob:
        gene.re_init(generate_funcs)

    elif r < (reinit_prob + smooth_param_prob):
        mutate_smooth_param3(gene)

    else:
        gene.tree = mutate_tree(gene.tree, gene.get_labels_not_in_tree())  # TODO add the labels here !!!


def mutate_gene(gene: GeneMain, param_prob=0.5):
    """
    Default strategy consists in modifying either a quantified parameter
    or the tree
    :param gene: the gene to be mutated
    :param param_prob: The probability to change a param, otherwise change the tree
    """
    if random.random() < param_prob:
        mutate_param(gene)

    else:
        gene.tree = mutate_tree(gene.tree, gene.get_labels_not_in_tree())  # TODO add the labels here !!!
        

def mutate_param2(gene):
    r = random.random()
    nb_param = 8
    if r < 1 / nb_param:
        gene.b = generate_random_b()
    elif r < 2 / nb_param:
        gene.m = generate_random_m()
    elif r < 3 / nb_param:
        gene.expr = generate_random_expr()
    elif r < 4 / nb_param:
        gene.deg = generate_random_deg()
    elif r < 5 / nb_param:
        gene.thr = generate_random_thr()
    elif r < 6 / nb_param:
        gene.init = generate_random_init()
    elif r < 7 / nb_param:
        gene.noise = generate_random_noise()
    else:
        gene.theta = generate_random_theta()
        
        
def mutate_param3(gene):
    r = random.random()
    nb_param = 7
    if r < 1 / nb_param:
        gene.b = generate_random_b()
    elif r < 2 / nb_param:
        gene.m = generate_random_m()
    elif r < 3 / nb_param:
        gene.expr = generate_random_expr2()
    elif r < 4 / nb_param:
        gene.deg = generate_random_deg2()
    elif r < 5 / nb_param:
        gene.init = generate_random_init()
    elif r < 6 / nb_param:
        gene.noise = generate_random_noise()
    else:
        gene.theta = generate_random_theta()


def mutate_smooth_param3(gene):
    r = random.uniform(0.8, 1.2)
    nb_param = 8
    if r < 1 / nb_param:
        gene.b = gene.b * r
    elif r < 2 / nb_param:
        gene.m = gene.m * r
    elif r < 3 / nb_param:
        gene.expr = gene.expr * r
    elif r < 4 / nb_param:
        gene.deg = gene.deg * r
    elif r < 5 / nb_param:
        gene.init = gene.init * r
    elif r < 6 / nb_param:
        gene.noise = gene.noise * r
    elif r < 7 / nb_param:
        gene.asym = gene.asym * r
    else:
        gene.theta = gene.theta * r


def mutate_param4(gene):
    r = random.random()
    nb_param = 8
    if r < 1 / nb_param:
        gene.b = generate_random_b()
    elif r < 2 / nb_param:
        gene.m = generate_random_m()
    elif r < 3 / nb_param:
        gene.expr = generate_random_expr2()
    elif r < 4 / nb_param:
        gene.deg = generate_random_deg2()
    elif r < 5 / nb_param:
        gene.init = generate_random_init()
    elif r < 6 / nb_param:
        gene.noise = generate_random_noise()
    elif r < 7 / nb_param:
        gene.asym = generate_random_asym()
    else:
        gene.theta = generate_random_theta()


def mutate_smooth_param4(gene):
    r = random.uniform(0.8, 1.2)
    nb_param = 7
    if r < 1 / nb_param:
        gene.b = gene.b * r
    elif r < 2 / nb_param:
        gene.m = gene.m * r
    elif r < 3 / nb_param:
        gene.expr = gene.expr * r
    elif r < 4 / nb_param:
        gene.deg = gene.deg * r
    elif r < 5 / nb_param:
        gene.init = gene.init * r
    elif r < 6 / nb_param:
        gene.noise = gene.noise * r
    else:
        gene.theta = gene.theta * r


def mutate_param(gene):
    r = random.random()
    nb_param = 6
    if r < 1 / nb_param:
        gene.b = generate_random_b()
    elif r < 2 / nb_param:
        gene.m = generate_random_m()
    elif r < 3 / nb_param:
        gene.expr = generate_random_expr()
    elif r < 4 / nb_param:
        gene.deg = generate_random_deg()
    elif r < 5 / nb_param:
        gene.thr = generate_random_thr()
    else:
        gene.theta = generate_random_theta()


def mutate_tree(tree: Formula, labels: set):
    if len(tree.labels_set()) >= 3:  # for grn with too many genes
        func = random.choice([tree_remove_gene, tree_substitute_gene])
    else:
        func = random.choice([tree_add_gene, tree_remove_gene, tree_substitute_gene])
    # make that cannot add if already too much genes
    try:
        new_tree = func(tree, labels)
    except AssertionError:
        new_tree = tree

    return new_tree


def tree_from_scratch(tree: Formula, labels: set) -> Formula:
    # TODO implicit initialization ...
    return generate_random_tree(labels | tree.labels_set())


def tree_add_gene(tree: Formula, labels: set) -> Formula:
    tree = tree.copy()
    op = random.choice(_op)
    available_labels = list(labels - tree.labels_set())
    assert len(available_labels) > 0, "Cannot substitute because no gene is available"
    gene = random.choice(available_labels)
    gene = Var(gene)

    # because the "not" does not take a second gene
    arg_gene = (gene,) if op.nb_children == 2 else ()

    target = tree.random_op()

    if target.is_root():
        tree = op(target, *arg_gene)

    else:
        target.parent.replace(target, op(target, *arg_gene))

    tree.update()
    return tree


def tree_substitute_gene(tree: Formula, labels: set) -> Formula:
    tree = tree.copy()
    available_labels = list(labels - tree.labels_set())
    assert len(available_labels) > 0, "Cannot substitute because no gene is available"
    gene = random.choice(available_labels)
    gene = Var(gene)

    target = tree.random_gene()

    if target.is_root():
        tree = gene

    else:
        target.parent.replace(target, gene)

    tree.update()
    return tree


def tree_remove_gene(tree: Formula, labels: set) -> Formula:
    tree = tree.copy()
    assert len(tree.labels_set()) > 1, "Cannot remove gene when there is only 1 gene"
    target = tree.random_gene()

    # find closer binary parent
    parent, direct_child = target.parent, target
    while parent.nb_children != 2:
        parent, direct_child = parent.parent, direct_child.parent

    other_child = parent.other_child(direct_child)

    if parent.is_root():
        tree = other_child

    else:
        parent.parent.replace(parent, other_child)

    tree.update()
    return tree
