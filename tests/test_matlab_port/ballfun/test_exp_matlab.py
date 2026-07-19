"""Port of MATLAB Chebfun tests/ballfun/test_exp.m (Fable 5).

FIXED (Fable 5): Ballfun.exp (compose) exercised by the port.

Provenance
----------
MATLAB source : tests/ballfun/test_exp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS

TOL = 1e2 * EPS


class TestBallfunExp:
    def test_all_matlab_assertions(self):
        # exp(z)
        f = Ballfun.from_function(lambda x, y, z: z).exp()
        exact = Ballfun.from_function(lambda x, y, z: jnp.exp(z))
        assert (f - exact).norm() < TOL
