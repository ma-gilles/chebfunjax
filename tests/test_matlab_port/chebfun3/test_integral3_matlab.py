"""Port of MATLAB Chebfun tests/chebfun3/test_integral3.m (Fable 5).

MATLAB integral3(f) == sum3(f); subdomain integral3(f, dom) is emulated
by constructing on the subdomain (sum3 has no domain argument).

Provenance
----------
MATLAB source : tests/chebfun3/test_integral3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS

TOL = 1e4 * EPS


class TestChebfun3Integral3:
    def test_odd_integrand(self):
        f = Chebfun3.from_function(lambda x, y, z: x * y * z)
        assert abs(float(f.sum3())) < TOL

    def test_polynomial_on_box(self):
        dom = (0.0, 3.0, -1.0, 1.0, 0.0, 1.0)
        f = Chebfun3.from_function(
            lambda x, y, z: x ** 2 * jnp.cos(y) + z, domain=dom)
        exact = 9 * 2 * np.sin(1.0) + 0.5 * 3 * 2
        assert abs(float(f.sum3()) - exact) < 10 * TOL
