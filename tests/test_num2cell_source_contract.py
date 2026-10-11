"""Native num2cell body controls; no upstream test_num2cell cohort exists."""

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix


@pytest.mark.parametrize("kind", ["scalar", "array", "quasi"])
@pytest.mark.parametrize("row", [False, True])
def test_source_dispatch(kind, row):
    a = cj.chebfun(lambda x: 1 + 2j + (3 - 4j) * x)
    if kind == "scalar":
        f = a.set_point_values([[7 + 8j], [9 - 10j]])
    elif kind == "array":
        f = cj.chebfun(lambda x: jnp.stack([1 + 2j + (3 - 4j) * x, 2 - x], axis=-1))
    else:
        f = Quasimatrix([a, 2 * a], a.domain)
    if row:
        f = f.transpose()
    cells = f.num2cell()
    n = len(cells)
    expected = f.mat2cell([1] * n, 1) if row else f.mat2cell(1, [1] * n)
    assert isinstance(cells, list)
    assert cells.shape == ((n, 1) if row else (1, n))
    for cell, reference in zip(cells, expected):
        assert cell.is_transposed == row
        np.testing.assert_array_equal(cell.chebcoeffs(), reference.chebcoeffs())
        np.testing.assert_array_equal(cell.point_values, reference.point_values)


def test_empty_identity():
    f = cj.chebfun([])
    cells = f.num2cell()
    assert cells == [f]
    assert cells[0] is f
    assert cells.shape == (1, 1)
