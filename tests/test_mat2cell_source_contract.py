"""Focused source API baseline, separate from original36 clauses."""

import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix


def test_quasimatrix_missing_api():
    f = cj.chebfun("x")
    q = Quasimatrix([f, 2 * f], f.domain)
    cells = q.mat2cell()
    assert len(cells) == 2 and cells[0] is f


def test_error_identifier():
    f = cj.Chebfun.horzcat(cj.chebfun("x"), cj.chebfun("x"))
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:mat2cell:size"):
        f.mat2cell(2, [1, 1])


def test_flat_list_complex_stored_values_and_orientation():
    import numpy as np

    f = cj.Chebfun.horzcat(cj.chebfun("x"), 2 * cj.chebfun("x"))
    f = f.set_point_values(np.asarray([[1 + 2j, 3 - 4j], [5 - 6j, 7 + 8j]]))
    before = np.asarray(f.point_values).copy()
    for source in [f, f.T]:
        cells = source.mat2cell()
        assert isinstance(cells, list) and len(list(iter(cells))) == 2
        assert cells.shape == ((2, 1) if source.is_transposed else (1, 2))
        for i, cell in enumerate(cells):
            assert cell.is_transposed == source.is_transposed
            np.testing.assert_array_equal(cell.point_values, before[:, i])
        joined = cj.Chebfun.horzcat(*[c.T if c.is_transposed else c for c in cells])
        np.testing.assert_array_equal(joined.point_values, before)
    np.testing.assert_array_equal(f.point_values, before)
