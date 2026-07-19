"""Port of MATLAB Chebfun tests/ballfunv/test_mrdivide.m (Fable 5).

FIXED (Fable 5): Ballfunv scalar division added in the audit.

Provenance
----------
MATLAB source : tests/ballfunv/test_mrdivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv

from ._helpers import EPS

TOL = 1e2 * EPS


class TestBallfunvMrdivide:
    def test_all_matlab_assertions(self):
        zero = Ballfun.from_function(lambda x, y, z: 0.0 * x)

        # Example 1: V/2
        f = Ballfun.from_function(lambda x, y, z: jnp.cos(y * z))
        V = Ballfunv(f, zero, zero)
        g = Ballfun.from_function(lambda x, y, z: jnp.cos(y * z) / 2)
        assert (V / 2 - Ballfunv(g, zero, zero)).norm() < TOL

        # Example 2: V/(-3)
        f = Ballfun.from_function(lambda x, y, z: jnp.sin(y * z))
        V = Ballfunv(f, f, f)
        g = Ballfun.from_function(lambda x, y, z: -jnp.sin(y * z) / 3)
        assert (V / (-3) - Ballfunv(g, g, g)).norm() < TOL

        # Example 3: V/1i
        f = Ballfun.from_function(lambda x, y, z: 2 * jnp.sin(x))
        V = Ballfunv(f, zero, f)
        g = Ballfun.from_function(lambda x, y, z: 2 * jnp.sin(x) / 1j)
        assert (V / 1j - Ballfunv(g, zero, g)).norm() < TOL
