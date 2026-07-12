"""Port of MATLAB Chebfun tests/chebfun2/test_mean.m (Fable 5).

FIXED: Chebfun2.mean/mean2 added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun2/test_mean.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2


class TestChebfun2Mean:
    def test_mean2(self):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        assert abs(float(f.mean2()) - float(f.sum2()) / 4) < 1e-14

    def test_mean_over_y(self):
        f = Chebfun2.from_function(lambda x, y: x + y * y)
        m = f.mean(dim=1)   # average over y: x + 1/3
        v = float(m(jnp.asarray(0.4), jnp.asarray(0.0)))
        assert abs(v - (0.4 + 1.0 / 3.0)) < 1e-12
