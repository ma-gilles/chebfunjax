"""Direct empty extraction follows the native extractColumns data path.

The pinned implementation slices the FUN coefficient columns and then slices
``pointValues(:, colIdx)`` while leaving the source orientation and domain in
place. These checks qualify the Python direct-method adapter, including the
empty-selector case.

Provenance
----------
MATLAB source : @chebfun/extractColumns.m, @chebtech/extractColumns.m
Chebfun commit: 7574c77
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj


def test_direct_empty_extract_columns_preserves_metadata():
    values = lambda x: jnp.stack((x, x**2, jnp.sin(x)), axis=-1)
    f = cj.chebfun(values, domain=(-2.0, 3.0)).set_point_values(
        jnp.array([[11.0, 12.0, 13.0], [21.0, 22.0, 23.0]]))

    empty_column = f.extract_columns([])
    assert empty_column.isempty()
    assert empty_column.domain == f.domain
    assert not empty_column.is_transposed
    assert empty_column.point_values.shape == (2, 0)

    empty_row = f.T.extract_columns([])
    assert empty_row.isempty()
    assert empty_row.domain == f.domain
    assert empty_row.is_transposed
    assert empty_row.point_values.shape == (2, 0)

    # Native extraction leaves F.isTransposed unchanged for nonempty columns too.
    row_column = f.T.extract_columns([1])
    assert not row_column.isempty()
    assert row_column.is_transposed
    assert row_column.domain == f.domain
    assert row_column.point_values.shape == (2, 1)
    assert jnp.array_equal(row_column.point_values[:, 0], jnp.array([12.0, 22.0]))

    # Empty-selector handling must not coerce malformed nonempty indices to int.
    with pytest.raises(TypeError):
        f.extract_columns([0.5])
    with pytest.raises(TypeError):
        f.extract_columns([True])
    with pytest.raises(IndexError, match="column index out of range"):
        f.extract_columns([3])
    with pytest.raises(IndexError, match="column index out of range"):
        f.extract_columns([-4])
    query = jnp.array([0.25])
    assert jnp.array_equal(f.extract_columns(-1)(query),
                           f.extract_columns(2)(query))
    assert f.extract_columns(np.int64(1)).n_columns == 1
    assert f.extract_columns(jnp.asarray(1, dtype=jnp.int32)).n_columns == 1

    scalar = f.extract_columns(0)
    scalar_again = scalar.extract_columns(0)
    assert jnp.array_equal(scalar_again.point_values, jnp.array([11.0, 21.0]))
    scalar_transposed = scalar.T.extract_columns([0])
    assert scalar_transposed.point_values.shape == (2, 1)
    assert jnp.array_equal(
        scalar_transposed.point_values[:, 0], jnp.array([11.0, 21.0]))

    scalar_empty = scalar.extract_columns([])
    assert scalar_empty.isempty()
    assert scalar_empty.domain == f.domain
    assert not scalar_empty.is_transposed
    assert scalar_empty.point_values.shape == (2, 0)

    # Source returns an already empty F unchanged before selecting columns.
    empty = cj.chebfun().T
    assert empty.extract_columns([]) is empty
