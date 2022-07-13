from pytest import fixture
import numpy as np
import jax.numpy as jnp

from lib.sde.cell.cell3 import Cell3, CellBatch3
from lib.sde.grn.grn3 import GRNMain3


@fixture
def grn():
    return GRNMain3(2, 0, 0)


@fixture
def cell(grn):
    return Cell3(grn)


def test_cell_step(cell):
    x = cell.quantities.copy()
    cell.run_step()
    assert (x != cell.quantities).any()


def run_test_cell_param(grn, id_to_check):
    # 1 is noise
    def_quant = jnp.array([0.5, 0.5])
    grn1 = grn.copy()
    grn1._params = np.array(grn1._params)
    grn1._params[1, :] = np.array([0, 0])
    grn1._params[id_to_check, :] = np.array([0, 0])

    grn2 = grn.copy()
    grn2._params = np.array(grn2._params)
    grn2._params[1, :] = np.array([0, 0])
    grn2._params[id_to_check, :] = np.array([1, 1])

    cell1 = Cell3(grn1)
    cell1.quantities = def_quant

    cell2 = Cell3(grn1)
    cell2.quantities = def_quant

    cell3 = Cell3(grn2)
    cell3.quantities = def_quant

    cell1.run_step(), cell2.run_step(), cell3.run_step()
    assert (cell1.quantities == cell2.quantities).all()
    assert not (cell1.quantities == cell3.quantities).all()


def test_cell_params(grn):
    for i in range(1, 7):
        run_test_cell_param(grn, 1)


def test_cell_step_batch(cell):
    x = cell.quantities.copy()
    batch = CellBatch3([cell])
    batch.run_step()
    assert (x != cell.quantities).any()


def run_test_cell_param_batch(grn, id_to_check):
    # 1 is noise
    def_quant = jnp.array([0.5, 0.5])
    grn1 = grn.copy()
    grn1._params = np.array(grn1._params)
    grn1._params[1, :] = np.array([0, 0])
    grn1._params[id_to_check, :] = np.array([0, 0])

    grn2 = grn.copy()
    grn2._params = np.array(grn2._params)
    grn2._params[1, :] = np.array([0, 0])
    grn2._params[id_to_check, :] = np.array([1, 1])

    cell1 = Cell3(grn1)
    cell1.quantities = def_quant
    batch1 = CellBatch3([cell1])

    cell2 = Cell3(grn1)
    cell2.quantities = def_quant
    batch2 = CellBatch3([cell2])

    cell3 = Cell3(grn2)
    cell3.quantities = def_quant
    batch3 = CellBatch3([cell3])

    batch1.run_step(), batch2.run_step(), batch3.run_step()
    assert (cell1.quantities == cell2.quantities).all()
    assert not (cell1.quantities == cell3.quantities).all()


def test_cell_params_batch(grn):
    for i in range(1, 7):
        run_test_cell_param_batch(grn, 1)

