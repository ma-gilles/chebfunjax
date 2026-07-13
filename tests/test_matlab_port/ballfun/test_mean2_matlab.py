"""Port of MATLAB Chebfun tests/ballfun/test_mean2.m (Fable 5).

FIXED: Ballfun.mean2(dims) added in the Fable 5 audit (average over
two spherical coordinates -> 1D Chebfun in the survivor; trig for
the lambda survivor).

Provenance
----------
MATLAB source : tests/ballfun/test_mean2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun

TOL = 1e-11
TS = jnp.asarray(np.linspace(0.05, 3.1, 9))
RS = jnp.asarray(np.linspace(0.0, 1.0, 9))


class TestBallfunMean2:
    def test_constant_means(self):
        # pass(1)-(2)
        f1 = Ballfun.from_function(lambda x, y, z: 1.0 + 0 * x)
        assert float(jnp.max(jnp.abs(
            f1.mean2((1, 2))(TS) - 1.0))) < TOL
        f2 = Ballfun.from_function(lambda x, y, z: 2.0 + 0 * x)
        assert float(jnp.max(jnp.abs(
            f2.mean2((2, 3))(RS) - 2.0))) < TOL

    def test_radial_profile(self):
        f = Ballfun.from_function(lambda r, lam, th: r ** 2,
                                  spherical=True)
        assert float(jnp.max(jnp.abs(
            f.mean2((2, 3))(RS) - RS ** 2))) < TOL
