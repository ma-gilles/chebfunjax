"""Port of MATLAB Chebfun tests/chebfun3/test_abs.m (Fable 5).

FIXED (Fable 5 audit): ``Chebfun3.abs()`` now exists.  For a positive
function ``abs(f) == f`` and ``abs(-f) == f``.

MATLAB pass(5) expects ``abs`` of a sign-changing chebfun3 to error;
chebfunjax's ``abs`` handles it without raising, so that case is not asserted.

Provenance
----------
MATLAB source : tests/chebfun3/test_abs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

EPS = float(np.finfo(np.float64).eps)
TOL = 1000 * EPS


class TestChebfun3Abs:
    def test_positive_default_domain(self):
        # pass(1,2): abs(f) == f and abs(-f) == f for f = cos(xyz) + 2 > 0.
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z) + 2)
        assert float((f - f.abs()).norm()) < TOL
        assert float((f - (-f).abs()).norm()) < TOL

    def test_positive_shifted_domain(self):
        # pass(3,4): same on [-3 4 -1 1 -2 0].
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z) + 2,
                                   domain=(-3, 4, -1, 1, -2, 0))
        assert float((f - f.abs()).norm()) < TOL
        assert float((f - (-f).abs()).norm()) < TOL
