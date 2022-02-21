from jax import jit


# here there is no distinction
# between a gene and a protein
def gene_expression(x, b, theta, m):
    x_ = (x + b)**m
    return x_ / (x_ + theta**m)


def gene_expression_bool(x, b, theta, m, thr):
    x_ = (x + b)**m
    return int((x_ / (x_ + theta**m)) > thr)


def gene_activation(x, b, theta, m):
    x_ = (x + b)**m
    return x_ / (x_ + theta**m)


def gene_derivation(expr, q, k1, k2):
    return k1 * expr - k2 * q


j_gene_expression = jit(gene_expression)
j_gene_expression_bool = jit(gene_expression_bool)
j_gene_activation = jit(gene_activation)
j_gene_derivation = jit(gene_derivation)
