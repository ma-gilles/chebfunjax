"""All eight native executed assertions; pass4 is assigned twice (seven slots).

Python explicit public concat calls replace MATLAB bracket syntax. Numeric
1-D operands represent columns for vertical concatenation.
"""

import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.operators.chebmatrix import ChebMatrix


def test_native_eight_executed_predicates():
    x = cj.chebfun("x", pref=ChebfunPref())
    f = cj.Chebfun.vertcat(x, 1)
    assert (
        isinstance(f, ChebMatrix)
        and f.size == (2, 1)
        and np.asarray(f.blocks[1][0]).dtype.kind in "biufc"
    )
    f = cj.Chebfun.vertcat(x, x)
    assert isinstance(f, ChebMatrix) and f.size == (2, 1) and isinstance(f.blocks[1][0], cj.Chebfun)
    row = cj.Chebfun.horzcat(x, 1)
    f = cj.Chebfun.vertcat(row, row)
    assert (
        isinstance(f, ChebMatrix)
        and f.size == (2, 2)
        and isinstance(f.blocks[1][0], cj.Chebfun)
        and isinstance(f.blocks[0][1], cj.Chebfun)
    )
    row = cj.Chebfun.horzcat(x, x)
    f = cj.Chebfun.vertcat(row, row)
    assert (
        isinstance(f, ChebMatrix)
        and f.size == (2, 2)
        and isinstance(f.blocks[1][0], cj.Chebfun)
        and isinstance(f.blocks[0][1], cj.Chebfun)
    )
    f = cj.Chebfun.vertcat(cj.Chebfun.horzcat(x, abs(x)), cj.Chebfun.horzcat(x, x))
    assert (
        isinstance(f, ChebMatrix)
        and f.size == (2, 2)
        and isinstance(f.blocks[1][0], cj.Chebfun)
        and isinstance(f.blocks[0][1], cj.Chebfun)
    )
    xt = x.ctranspose()
    f = cj.Chebfun.vertcat(xt, xt)
    assert isinstance(f, cj.Chebfun) and f.n_columns == 2
    with pytest.raises(Exception):
        cj.Chebfun.vertcat(x, xt)
    f = cj.Chebfun.vertcat(x, np.asarray([[0.0], [0.0]]))
    assert isinstance(f, ChebMatrix) and f.size == (3, 1)


def test_row_array_storage_without_refit_and_container_adapter():
    x = cj.chebfun("x")
    f = cj.Chebfun.horzcat(x, 2 * x).T
    # Native numElements>1 gives scalar ChebMatrix rows, not callback refit.
    out = cj.Chebfun.vertcat(f, f)
    assert isinstance(out, ChebMatrix) and out.size == (4, 1)
    for actual, expected in zip([row[0] for row in out.blocks], f.mat2cell() + f.mat2cell()):
        assert actual.is_transposed
        for a, b in zip(actual.funs, expected.funs):
            np.testing.assert_array_equal(a.tech.coeffs, b.tech.coeffs)
    row = cj.Chebfun.vertcat([x.T, (2 * x).T])
    assert row.is_transposed and row.n_columns == 2
    np.testing.assert_array_equal(
        row.point_values, np.column_stack([x.point_values, (2 * x).point_values])
    )


def test_native_row_branch_literal_final_conjugate_transpose():
    f = cj.Chebfun.from_coeffs(np.asarray([1 + 2j, 3 - 4j])).T
    g = cj.Chebfun.from_coeffs(np.asarray([2 - 1j, 4 + 5j])).T
    out = cj.Chebfun.vertcat(f, g)
    expected = cj.Chebfun.horzcat(f.T, g.T).ctranspose()
    assert out.is_transposed
    for a, b in zip(out.funs, expected.funs):
        np.testing.assert_array_equal(a.tech.coeffs, b.tech.coeffs)
    np.testing.assert_array_equal(out.point_values, expected.point_values)
