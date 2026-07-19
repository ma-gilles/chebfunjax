"""Port of MATLAB Chebfun tests/ballfun/test_tan.m (Fable 5).

FIXED (Fable 5): Ballfun.tan added in the audit.

Provenance
----------
MATLAB source : tests/ballfun/test_tan.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS

TOL = 1e4 * EPS


class TestBallfunTan:
    def test_all_matlab_assertions(self):
        f = Ballfun.from_function(lambda x, y, z: jnp.sin(y)).tan()
        exact = Ballfun.from_function(
            lambda x, y, z: jnp.tan(jnp.sin(y)))
        assert (f - exact).norm() < TOL
