"""Exact matrices for source rank-budget termination.

Provenance
----------
MATLAB source : @chebfun2/constructor.m (completeACA)
Chebfun commit: 7574c77
"""
import numpy as np
import pytest

from chebfunjax.chebfun2d.separable_approx import _complete_aca


@pytest.mark.parametrize("size,rank,factor,expected", [
    (8, 2, 4, True), (8, 2, 2, False),
    (9, 3, 4, True), (9, 2, 4, False),
])
def test_exact_residual_keeps_source_rank_budget(size, rank, factor, expected):
    matrix = np.zeros((size, size))
    for k in range(rank):
        matrix[k, k] = 2.0 ** (-k)
    pivots, positions, rows, cols, ifail = _complete_aca(matrix, 1e-12, factor)
    assert len(pivots) == rank
    assert ifail is expected
    assert np.array_equal(positions, np.column_stack((np.arange(rank), np.arange(rank))))
    assert np.array_equal(cols @ np.diag(1 / pivots) @ rows, matrix)


def test_zero_early_return_stays_resolved():
    pivots, _, _, _, ifail = _complete_aca(np.zeros((8, 8)), 1e-12, 4)
    assert np.array_equal(pivots, np.array([0.]))
    assert ifail is False
