"""Port of MATLAB Chebfun tests/ballfun/test_sin.m (Fable 5).

FIXED (Fable 5): Ballfun.sin (compose) exercised by the port.

Provenance
----------
MATLAB source : tests/ballfun/test_sin.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS

TOL = 1e4 * EPS


class TestBallfunSin:
    def test_all_matlab_assertions(self):
        f = Ballfun.from_function(lambda x, y, z: x * y * z).sin()
        exact = Ballfun.from_function(
            lambda x, y, z: jnp.sin(x * y * z))
        assert (f - exact).norm() < TOL
