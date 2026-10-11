"""Port of MATLAB Chebfun tests/ballfun/test_times.m (Fable 5).

FIXED (Fable 5): Ballfun*Ballfun product exercised by the port.

Provenance
----------
MATLAB source : tests/ballfun/test_times.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS, X0, Y0, val

TOL = 1e4 * EPS


class TestBallfunTimes:
    def test_all_matlab_assertions(self):
        # Constants: 2 * 3 = 6.
        f = Ballfun.from_values(2*jnp.ones((21, 12, 22)))
        g = Ballfun.from_values(3*jnp.ones((23, 20, 24)))
        h = Ballfun.from_values(6*jnp.ones((25, 20, 10)))
        assert (f * g - h).norm() < TOL
        assert (g * f - h).norm() < TOL

    def test_pointwise_product(self):
        f = Ballfun.from_function(lambda x, y, z: x)
        g = Ballfun.from_function(lambda x, y, z: y)
        assert abs(val(f * g) - X0 * Y0) < 1e3 * EPS

    def test_scalar_times(self):
        f = Ballfun.from_function(lambda x, y, z: x)
        assert abs(val(3 * f) - 3 * X0) < 1e3 * EPS
        assert abs(val(f * 3) - 3 * X0) < 1e3 * EPS
