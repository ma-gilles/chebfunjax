"""Literal phase-two GE arithmetic, Chebfun 7574c77 @chebfun2/constructor.m."""
import numpy as np
import pytest

from chebfunjax.chebfun2d.separable_approx import _ge_on_skeleton


@pytest.mark.parametrize("rank", [1, 2, 3, 5])
def test_skeleton_source_operation_order(rank):
    rng = np.random.RandomState(103 + rank)
    columns = rng.randn(9, rank)
    rows = rng.randn(rank, 11)
    pivots = rng.randn(rank) + 1.123
    positions = np.column_stack([np.arange(rank), np.arange(rank)])
    expected_columns, expected_rows = columns.copy(), rows.copy()
    for k in range(rank - 1):
        expected_columns[:, k+1:] -= np.outer(
            expected_columns[:, k],
            expected_rows[k, positions[k+1:, 1]] / pivots[k])
        expected_rows[k+1:, :] -= np.outer(
            expected_columns[positions[k+1:, 0], k],
            expected_rows[k, :] / pivots[k])
    actual_columns, actual_rows = _ge_on_skeleton(columns, rows, pivots, positions)
    np.testing.assert_array_equal(actual_columns, expected_columns)
    np.testing.assert_array_equal(actual_rows, expected_rows)
