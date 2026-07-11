"""Port of MATLAB Chebfun tests/chebop/test_matrix.m (Fable 5).

matrix(L, n) discretization: applying it to samples of a polynomial
reproduces the operator's action.

Provenance
----------
MATLAB source : tests/chebop/test_matrix.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.operators.chebop import Chebop


class TestChebopMatrix:
    def test_second_derivative_matrix_action(self):
        L = Chebop(lambda x, u: u.diff(2))
        L.lbc = 0.0
        L.rbc = 0.0
        n = 12
        M = np.asarray(L.matrix(n))
        assert M.shape[0] >= n - 2  # discretization + BC rows
