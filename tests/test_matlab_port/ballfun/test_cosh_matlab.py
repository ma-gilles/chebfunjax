"""Port of MATLAB Chebfun tests/ballfun/test_cosh.m (Fable 5).

FIXED (Fable 5): Ballfun.cosh added in the audit.

Provenance
----------
MATLAB source : tests/ballfun/test_cosh.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS

TOL = 1e2 * EPS


class TestBallfunCosh:
    def test_all_matlab_assertions(self):
        f = Ballfun.from_function(lambda x, y, z: y).cosh()
        exact = Ballfun.from_function(lambda x, y, z: jnp.cosh(y))
        assert (f - exact).norm() < TOL
