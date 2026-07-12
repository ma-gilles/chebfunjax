"""Port of MATLAB Chebfun tests/chebfun2/test_minandmax2est.m (Fable 5).

FIXED: Chebfun2.minandmax2 added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun2/test_minandmax2est.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2


class TestChebfun2Minandmax2est:
    def test_extrema_of_quadratic(self):
        g = Chebfun2.from_function(
            lambda x, y: 1 - (x - 0.2) ** 2 - (y + 0.1) ** 2)
        vals, locs = g.minandmax2()
        assert abs(float(vals[1]) - 1.0) < 1e-10
        np.testing.assert_allclose(np.asarray(locs[1]), [0.2, -0.1],
                                   atol=1e-6)
        assert abs(float(vals[0]) - (1 - 1.2 ** 2 - 1.1 ** 2)) < 1e-10
