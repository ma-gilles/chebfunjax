"""Port of MATLAB Chebfun tests/ballfun/test_tanh.m (Fable 5).

FIXED (Fable 5): Ballfun.tanh added in the audit.

Provenance
----------
MATLAB source : tests/ballfun/test_tanh.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS

TOL = 1e4 * EPS


class TestBallfunTanh:
    def test_all_matlab_assertions(self):
        f = Ballfun.from_function(lambda x, y, z: jnp.cos(x * z)).tanh()
        exact = Ballfun.from_function(
            lambda x, y, z: jnp.tanh(jnp.cos(x * z)))
        assert (f - exact).norm() < TOL
