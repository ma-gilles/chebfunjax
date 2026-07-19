"""Port of MATLAB Chebfun tests/ballfunv/test_power.m (Fable 5).

FIXED (Fable 5): Ballfunv.power exercised by the port.

Provenance
----------
MATLAB source : tests/ballfunv/test_power.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv

from ._helpers import EPS

TOL = 1e4 * EPS


class TestBallfunvPower:
    def test_all_matlab_assertions(self):
        f = Ballfun.from_function(lambda x, y, z: jnp.cos(x * y) * z)
        zero = Ballfun.from_function(lambda x, y, z: 0.0 * x)
        one = Ballfun.from_function(lambda r, lam, th: 1.0 + 0.0 * r,
                                    spherical=True)

        # Example 1: power 2 of (f, 0, 0).
        F = Ballfunv(f, zero, zero)
        H = F.power(2)
        Hexact = Ballfunv(f ** 2, zero, zero)
        assert (H - Hexact).norm() < TOL

        # Example 2: power 0 of any field is (1, 1, 1).
        F = Ballfunv(f, 2 * f, 3 * f)
        H = F.power(0)
        assert (H - Ballfunv(one, one, one)).norm() < TOL

        # Example 3: power 3 componentwise.
        F = Ballfunv(f, 2 * f, 3 * f)
        H = F.power(3)
        Hexact = Ballfunv(f ** 3, 8 * (f ** 3), 27 * (f ** 3))
        assert (H - Hexact).norm() < TOL
