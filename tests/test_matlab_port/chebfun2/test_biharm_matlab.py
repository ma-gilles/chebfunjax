"""Port of MATLAB Chebfun tests/chebfun2/test_biharm.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): ``Chebfun2.biharm()`` /
``biharmonic()`` now exist.

Provenance
----------
MATLAB source : tests/chebfun2/test_biharm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 1e6 * EPS


class TestChebfun2Biharm:
    def test_biharmonic_of_x2y2(self):
        # The biharmonic operator applied to x^2 y^2 is the constant 8
        # (f_xxxx + f_yyyy + 2 f_xxyy = 0 + 0 + 2*4).
        f = Chebfun2.from_function(lambda x, y: x ** 2 * y ** 2)
        g = Chebfun2.from_function(lambda x, y: 8.0 + 0 * x)
        assert float((f.biharm() - g).norm()) < TOL

    def test_biharm_is_biharmonic(self):
        # biharm is documented shorthand for biharmonic.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        assert float((f.biharm() - f.biharmonic()).norm()) < TOL
