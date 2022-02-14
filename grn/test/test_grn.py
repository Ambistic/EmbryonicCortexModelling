import pickle

from lib.sde.grn.grn import GRNSingular, GRNOpt, GRNMain
from lib.sde.grn.grn2 import GRNMain2
from lib.sde.cell.cell import Cell


def test_grn():
    grn = GRNSingular(3, 0, 3)
    print(grn)
    print([gene.tree for gene in grn.genes])
    assert True


def test_run_grn():
    print()
    grn = GRNSingular(3, 0, 3)
    grn.print_trees()
    grn.print_params()
    for i in range(10):
        print()
        grn.run_step()
        grn.print_quant()
    assert True


def test_run_grnopt():
    print()
    grn = GRNOpt(5, 0, 3)
    grn.print_trees()
    grn.print_params()
    for i in range(10):
        print()
        grn.run_step()
        grn.print_quant()
    assert True


def test_json_grnopt():
    grn = GRNOpt(5, 0, 3)
    print(grn.to_json())


def test_copy_grnopt():
    grn = GRNOpt(5, 0, 3)
    grn2 = grn.copy()
    assert (grn._b == grn2._b).all()
    

def test_copy_grn_main():
    grn = GRNMain(5, 0, 3)
    grn2 = grn.copy()
    assert (grn._b == grn2._b).all()


def test_cell():
    grn = GRNMain(5, 0, 3)
    cell = Cell(grn)
    cell.print_gene_info(1)
    for i in range(10):
        cell.run_step()
    cell.print_gene_info(1)


def test_cell_division():
    grn = GRNMain(5, 0, 3)
    cell = Cell(grn)
    for i in range(10):
        cell.run_step()
        print()
    print(cell.quantities)
    cell1, cell2 = cell.divide()
    print(cell1.quantities)
    print(cell2.quantities)


def test_copy_grn_main_2():
    grn = GRNMain2(5, 0, 3)
    grn2 = grn.copy()
    assert (grn._params == grn2._params).all()


def test_immutable_copy_grn_main_2():
    grn = GRNMain2(5, 0, 3)
    grn2 = grn.copy()
    grn2.set_mutable()
    grn2.genes[0].b = grn.genes[0].b - 1
    grn2.compile()
    print()
    print(grn2)
    print(grn)
    assert grn2.genes[0].b == grn.genes[0].b - 1


def test_grn_access():
    grn = GRNMain2(5, 0, 3)
    assert (grn.init, grn.noise, grn.b, grn.m, grn.theta, grn.deg, grn.thr, grn.expr)


def test_grn_main_2_pickle():
    grn = GRNMain2(5, 0, 3)
    pck = pickle.dumps(grn)
    grn2 = pickle.loads(pck)
    assert (grn._params == grn2._params).all()
