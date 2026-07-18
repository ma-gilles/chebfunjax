"""Port of MATLAB Chebfun tests/chebfun2/test_abs.m (Fable 5).

FIXED (Fable 5 audit): ``Chebfun2.abs()`` now exists.  For a positive
function ``abs(f) == f`` and ``abs(-f) == f``.

Provenance
----------
MATLAB source : tests/chebfun2/test_abs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 1000 * EPS


class TestChebfun2Abs:
    def test_positive_default_domain(self):
        # pass(1,2): abs(f) == f and abs(-f) == f for f = cos(xy) + 2 > 0.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y) + 2)
        assert float((f - f.abs()).norm()) < TOL
        assert float((f - (-f).abs()).norm()) < TOL

    def test_positive_shifted_domain(self):
        # pass(3,4): same on [-3 4 -1 10].
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y) + 2,
                                   domain=(-3, 4, -1, 10))
        assert float((f - f.abs()).norm()) < TOL
        assert float((f - (-f).abs()).norm()) < TOL
