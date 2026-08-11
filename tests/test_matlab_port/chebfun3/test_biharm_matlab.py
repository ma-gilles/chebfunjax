"""Port of MATLAB Chebfun tests/chebfun3/test_biharm.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): ``Chebfun3.biharm()`` /
``biharmonic()`` now exist.

Provenance
----------
MATLAB source : tests/chebfun3/test_biharm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3

EPS = float(np.finfo(np.float64).eps)
TOL = 1e6 * EPS


class TestChebfun3Biharm:
    def test_biharmonic_of_quadratic_products(self):
        # biharm(x^2 y^2 + x^2 z^2 + y^2 z^2) = 2*(4 + 4 + 4) = 24.
        f = chebfun3(lambda x, y, z:
                     x ** 2 * y ** 2 + x ** 2 * z ** 2 + y ** 2 * z ** 2)
        g = chebfun3(lambda x, y, z: 24.0 + 0 * x)
        assert float((f.biharm() - g).norm()) < TOL

    def test_biharm_is_biharmonic(self):
        # biharm is documented shorthand for biharmonic.
        f = chebfun3(lambda x, y, z: x ** 2 * y ** 2)
        assert float((f.biharm() - f.biharmonic()).norm()) < TOL
