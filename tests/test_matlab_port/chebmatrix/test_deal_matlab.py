"""Port of MATLAB Chebfun tests/chebmatrix/test_deal.m (Fable 5).

FIXED: ChebMatrix container API added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebmatrix/test_deal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.operators.chebmatrix import ChebMatrix


class TestChebmatrixDeal:
    def test_deal(self):
        a = ChebMatrix.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
        blks = a.deal()
        assert blks == [1.0, 2.0, 3.0, 4.0]
