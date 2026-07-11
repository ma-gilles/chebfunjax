"""Port of MATLAB Chebfun tests/diskfun/test_integral2.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_integral2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

TOL = 1e-10


class TestDiskfunIntegral2:
    def test_odd_integrand_zero(self):
        f = Diskfun.from_function(lambda t, r: r * jnp.cos(t))
        assert abs(float(f.sum())) < 100 * TOL

    def test_polynomial_moment(self):
        # int_disk (x^2 + y^2) = int r^2 * r dr dt = pi/2
        f = Diskfun.from_function(lambda t, r: r ** 2)
        assert abs(float(f.sum()) - np.pi / 2) < 100 * TOL
