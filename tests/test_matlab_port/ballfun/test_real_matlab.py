"""Port of MATLAB Chebfun tests/ballfun/test_real.m (Fable 5).

FIXED (Fable 5): Ballfun.real added in the audit.

Provenance
----------
MATLAB source : tests/ballfun/test_real.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS

TOL = 1e4 * EPS


class TestBallfunReal:
    def test_all_matlab_assertions(self):
        # Example 1: real(x + 1i*y) = x
        f = Ballfun.from_function(lambda x, y, z: x + 1j * y).real()
        exact = Ballfun.from_function(lambda x, y, z: x)
        assert (f - exact).norm() < TOL

        # Example 2: real(sin(z) + 1i*cos(y)) = sin(z)
        f = Ballfun.from_function(
            lambda x, y, z: jnp.sin(z) + 1j * jnp.cos(y)).real()
        exact = Ballfun.from_function(lambda x, y, z: jnp.sin(z))
        assert (f - exact).norm() < TOL
