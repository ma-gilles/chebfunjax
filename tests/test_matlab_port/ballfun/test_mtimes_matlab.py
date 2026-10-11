"""Port of MATLAB Chebfun tests/ballfun/test_mtimes.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_mtimes.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS, X0, val


class TestBallfunMtimes:
    def test_scalar_mtimes(self):
        f = Ballfun.from_function(lambda x, y, z: x)
        assert abs(val(0.5 * f) - 0.5 * X0) < 1e3 * EPS


    def test_native_numeric_grid_norms(self):
        f = Ballfun.from_values(jnp.ones((21, 20, 22)))
        g = Ballfun.from_values(2*jnp.ones((21, 20, 22)))
        assert (2*f - g).norm() < 1e4*EPS
        assert (f*2 - g).norm() < 1e4*EPS
