"""Port of MATLAB Chebfun tests/chebmatrix/test_flip.m (Fable 5).

FIXED: ChebMatrix container API added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebmatrix/test_flip.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.operators.chebmatrix import ChebMatrix


class TestChebmatrixFlip:
    def test_fliplr_flipud(self):
        a = ChebMatrix.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert a.fliplr()[0, 0] == 2.0
        assert a.flipud()[0, 0] == 3.0
