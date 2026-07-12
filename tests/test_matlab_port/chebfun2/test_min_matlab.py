"""Port of MATLAB Chebfun tests/chebfun2/test_min.m (Fable 5).

FIXED: Chebfun2.min2 added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun2/test_min.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2


class TestChebfun2Min:
    def test_min2(self):
        g = Chebfun2.from_function(
            lambda x, y: (x - 0.1) ** 2 + (y - 0.2) ** 2)
        v, loc = g.min2()
        assert abs(float(v)) < 1e-10
        np.testing.assert_allclose(np.asarray(loc), [0.1, 0.2],
                                   atol=1e-5)
