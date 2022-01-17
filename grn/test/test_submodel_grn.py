from model import Action
from submodels.grn_v1 import GRNModelFactory
from lib.sde.grn import GRNMain
import pytest


@pytest.fixture()
def submodel_grn1():
    grn = GRNMain(5, 0, 5)
    factory = GRNModelFactory(grn=grn)

    one_cell = factory.generate(0)

    for i in range(100):
        one_cell.tick(i, 0.1)

    return one_cell


def test_grn1_diff(submodel_grn1):
    cell = submodel_grn1
    cell.cell_program.quantities[2] = 0
    assert cell._read_cell_program() != Action.DiffNeuron

    cell.cell_program.quantities[2] = 10
    assert cell._read_cell_program() == Action.DiffNeuron


def test_grn1_div(submodel_grn1):
    cell = submodel_grn1
    cell.cell_program.quantities[2] = 0
    cell.cell_program.quantities[1] = 0
    assert cell._read_cell_program() != Action.Divide

    cell.cell_program.quantities[1] = 10
    assert cell._read_cell_program() == Action.Divide
