"""Port of MATLAB Chebfun tests/ballfun/test_power.m (Fable 5).

FIXED (Fable 5): Ballfun.__pow__ exercised by the port.

Provenance
----------
MATLAB source : tests/ballfun/test_power.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS

TOL = 1e4 * EPS


class TestBallfunPower:
    def test_all_matlab_assertions(self):
        # (r cos(lam) sin(th))^5 (spherical)
        f = Ballfun.from_function(
            lambda r, lam, th: r * jnp.cos(lam) * jnp.sin(th),
            spherical=True)
        g = f ** 5
        h = Ballfun.from_function(
            lambda r, lam, th: r ** 5 * jnp.cos(lam) ** 5
            * jnp.sin(th) ** 5, spherical=True)
        assert (g - h).norm() < TOL
