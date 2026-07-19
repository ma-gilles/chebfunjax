"""Port of MATLAB Chebfun tests/ballfunv/test_mtimes.m (Fable 5).

FIXED (Fable 5): Ballfunv scalar and scalar-field multiplication
exercised by the port.

Provenance
----------
MATLAB source : tests/ballfunv/test_mtimes.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv

from ._helpers import EPS

TOL = 1e2 * EPS


class TestBallfunvMtimes:
    def test_all_matlab_assertions(self):
        # Scalar times: 2*F and F*2.
        f = Ballfun.from_function(lambda x, y, z: 1.0 + 0.0 * x)
        F = Ballfunv(f, f, f)
        g = Ballfun.from_function(lambda x, y, z: 2.0 + 0.0 * x)
        G = Ballfunv(g, g, g)
        assert (2 * F - G).norm() < TOL
        assert (F * 2 - G).norm() < TOL

        # Scalar-field times a vector: V.*f and f.*V.
        V = Ballfunv(
            Ballfun.from_function(lambda x, y, z: x),
            Ballfun.from_function(lambda x, y, z: y),
            Ballfun.from_function(lambda x, y, z: z))
        fc = Ballfun.from_function(lambda x, y, z: jnp.cos(y))
        exact = Ballfunv(
            Ballfun.from_function(lambda x, y, z: x * jnp.cos(y)),
            Ballfun.from_function(lambda x, y, z: y * jnp.cos(y)),
            Ballfun.from_function(lambda x, y, z: z * jnp.cos(y)))
        assert (V * fc - exact).norm() < TOL
        assert (fc * V - exact).norm() < TOL
