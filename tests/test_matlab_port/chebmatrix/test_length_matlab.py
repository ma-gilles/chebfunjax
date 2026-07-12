"""Port of MATLAB Chebfun tests/chebmatrix/test_length.m (Fable 5).

FIXED: ChebMatrix container API added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebmatrix/test_length.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import chebfunjax as cj
from chebfunjax.operators.chebmatrix import ChebMatrix


class TestChebmatrixLength:
    def test_length(self):
        x = cj.chebfun(lambda t: t)
        ff = ChebMatrix.from_array([[x, x, x]])
        assert len(ff) == 3
        assert len(ff.T) == 3
