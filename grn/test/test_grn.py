from lib.sde.grn import GRN


def test_grn():
    grn = GRN(3, 0, 3)
    print(grn)
    print([gene.tree for gene in grn.genes])
    assert True


def test_run_grn():
    print()
    grn = GRN(3, 0, 3)
    grn.print_trees()
    grn.print_params()
    for i in range(10):
        print()
        grn.run_step()
        grn.print_quant()
    assert True
    
