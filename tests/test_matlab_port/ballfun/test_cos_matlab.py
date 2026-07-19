"""Port of MATLAB Chebfun tests/ballfun/test_cos.m (Fable 5).

FIXED (Fable 5): Ballfun.cos (compose) exercised by the port.

Provenance
----------
MATLAB source : tests/ballfun/test_cos.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS

TOL = 1e2 * EPS


class TestBallfunCos:
    def test_all_matlab_assertions(self):
        f = Ballfun.from_function(lambda x, y, z: x).cos()
        exact = Ballfun.from_function(lambda x, y, z: jnp.cos(x))
        assert (f - exact).norm() < TOL
