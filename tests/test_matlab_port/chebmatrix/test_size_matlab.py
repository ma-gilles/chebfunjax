"""Port of MATLAB Chebfun tests/chebmatrix/test_size.m (Fable 5).

FIXED: ChebMatrix container API added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebmatrix/test_size.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import chebfunjax as cj
from chebfunjax.operators.chebmatrix import ChebMatrix


class TestChebmatrixSize:
    def test_sizes_and_transpose(self):
        x = cj.chebfun(lambda t: t)
        ff = ChebMatrix.from_array([[x, x, x]])
        assert ff.size == (1, 3)
        assert ff.T.size == (3, 1)
