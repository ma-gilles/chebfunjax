"""Port of MATLAB Chebfun tests/chebmatrix/test_norm.m (Fable 5).

FIXED: ChebMatrix container API added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebmatrix/test_norm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

import chebfunjax as cj
from chebfunjax.operators.chebmatrix import ChebMatrix


class TestChebmatrixNorm:
    def test_norm(self):
        x = cj.chebfun(lambda t: t)
        ff = ChebMatrix.from_array([[x, x, x]])
        # ||x||_2^2 = 2/3 per block
        assert abs(ff.norm() - np.sqrt(2.0)) < 1e-14
