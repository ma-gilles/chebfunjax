"""Port of MATLAB Chebfun tests/chebfun2/test_integral2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_integral2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


class TestChebfun2Integral2:
    def test_odd_integrand_zero(self):
        f = Chebfun2.from_function(lambda x, y: x * y)
        assert abs(float(f.sum2())) < TOL

    def test_subdomain_integral(self):
        # MATLAB: integral2(f, [0 1 0 1]) == 1/4.  chebfunjax sum2 has no
        # subdomain argument; integrate a fresh construction instead.
        f = Chebfun2.from_function(lambda x, y: x * y,
                                   domain=(0.0, 1.0, 0.0, 1.0))
        assert abs(float(f.sum2()) - 0.25) < TOL

    def test_stretched_domain(self):
        f = Chebfun2.from_function(lambda x, y: x ** 2 * jnp.cos(y),
                                   domain=(0.0, 3.0, -1.0, 1.0))
        assert abs(float(f.sum2()) - 9 * 2 * np.sin(1.0)) < TOL
