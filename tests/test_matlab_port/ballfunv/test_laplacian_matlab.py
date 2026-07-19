"""Port of MATLAB Chebfun tests/ballfunv/test_laplacian.m (Fable 5).

FIXED (Fable 5): Ballfunv.laplacian added in the audit (componentwise).

Provenance
----------
MATLAB source : tests/ballfunv/test_laplacian.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv

from ._helpers import EPS

TOL = 1e7 * EPS


class TestBallfunvLaplacian:
    def test_all_matlab_assertions(self):
        # Example 1: lap of (r^2, 2 r^2, -r^2) is the constant (6, 12, -6).
        f = Ballfun.from_function(lambda x, y, z: x ** 2 + y ** 2 + z ** 2)
        V = Ballfunv(f, 2 * f, -f)
        g = V.laplacian()
        exact = Ballfunv(
            Ballfun.from_function(lambda x, y, z: 6.0 + 0.0 * x),
            Ballfun.from_function(lambda x, y, z: 12.0 + 0.0 * x),
            Ballfun.from_function(lambda x, y, z: -6.0 + 0.0 * x))
        assert (g - exact).norm() < TOL

        # Example 2: vector identity lap(V) = grad(div V) - curl(curl V).
        f1 = Ballfun.from_function(lambda x, y, z: jnp.cos(x * y))
        f2 = Ballfun.from_function(lambda x, y, z: jnp.sin(y * z))
        f3 = Ballfun.from_function(lambda x, y, z: z ** 3)
        v = Ballfunv(f1, f2, f3)
        g = v.laplacian()
        gd = v.div().grad()
        gdv = Ballfunv(*(gd.components if hasattr(gd, "components")
                         else list(gd)))
        exact = gdv - v.curl().curl()
        assert (g - exact).norm() < TOL
