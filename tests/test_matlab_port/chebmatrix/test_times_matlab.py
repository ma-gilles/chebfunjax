"""Port of MATLAB Chebfun tests/chebmatrix/test_times.m (Fable 5).

FIXED: ChebMatrix container API added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebmatrix/test_times.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.operators.chebmatrix import ChebMatrix


class TestChebmatrixTimes:
    def test_elementwise_times(self):
        a = ChebMatrix.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = a.times(a)
        assert b[1, 1] == 16.0
        assert a.times(2.0)[0, 1] == 4.0
