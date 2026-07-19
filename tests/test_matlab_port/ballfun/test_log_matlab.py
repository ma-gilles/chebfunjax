"""Port of MATLAB Chebfun tests/ballfun/test_log.m (Fable 5).

FIXED (Fable 5): Ballfun.log added in the audit.

Provenance
----------
MATLAB source : tests/ballfun/test_log.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS

TOL = 1e4 * EPS


class TestBallfunLog:
    def test_all_matlab_assertions(self):
        # Example 1: log(10) constant (spherical)
        f = Ballfun.from_function(
            lambda r, lam, th: 10.0 + 0.0 * r, spherical=True).log()
        exact = Ballfun.from_function(
            lambda r, lam, th: np.log(10.0) + 0.0 * r, spherical=True)
        assert (f - exact).norm() < TOL

        # Example 2: log(exp(y)) = y
        f = Ballfun.from_function(lambda x, y, z: jnp.exp(y)).log()
        exact = Ballfun.from_function(lambda x, y, z: y)
        assert (f - exact).norm() < TOL

        # Example 3: log(exp(x*z)) = x*z
        f = Ballfun.from_function(lambda x, y, z: jnp.exp(x * z)).log()
        exact = Ballfun.from_function(lambda x, y, z: x * z)
        assert (f - exact).norm() < TOL
