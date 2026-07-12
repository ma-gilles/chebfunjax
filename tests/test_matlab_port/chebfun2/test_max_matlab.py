"""Port of MATLAB Chebfun tests/chebfun2/test_max.m (Fable 5).

FIXED: Chebfun2.max2 added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun2/test_max.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2


class TestChebfun2Max:
    def test_max2(self):
        g = Chebfun2.from_function(
            lambda x, y: jnp.cos(2 * x) * jnp.cos(y))
        v, loc = g.max2()
        assert abs(float(v) - 1.0) < 1e-10
