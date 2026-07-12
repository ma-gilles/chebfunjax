"""Port of MATLAB Chebfun tests/chebmatrix/test_constructor.m (Fable 5).

FIXED: ChebMatrix container API added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebmatrix/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.operators.chebmatrix import ChebMatrix


class TestChebmatrixConstructor:
    def test_from_array_equals_from_cells(self):
        r = np.random.default_rng(42).random((2, 2))
        A = ChebMatrix.from_array(r)
        B = ChebMatrix.from_array(
            [[float(r[i, j]) for j in range(2)] for i in range(2)])
        assert (A - B).norm() < 1e-14
