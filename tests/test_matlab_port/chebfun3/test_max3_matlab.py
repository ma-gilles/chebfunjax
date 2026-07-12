"""Port of MATLAB Chebfun tests/chebfun3/test_max3.m (Fable 5).

FIXED: Chebfun3.max3 added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_max3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


class TestChebfun3Max3:
    def test_max3_of_quadratic(self):
        g = Chebfun3.from_function(
            lambda x, y, z: 1 - (x - 0.2) ** 2 - (y + 0.1) ** 2
            - z ** 2)
        v, loc = g.max3()
        assert abs(float(v) - 1.0) < 1e-8
        np.testing.assert_allclose(np.asarray(loc), [0.2, -0.1, 0.0],
                                   atol=1e-4)
