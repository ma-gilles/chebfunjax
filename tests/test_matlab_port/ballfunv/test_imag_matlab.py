"""Port of MATLAB Chebfun tests/ballfunv/test_imag.m (Fable 5).

FIXED (Fable 5): Ballfunv.imag added in the audit (componentwise).

Provenance
----------
MATLAB source : tests/ballfunv/test_imag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv

from ._helpers import EPS

TOL = 1e2 * EPS


class TestBallfunvImag:
    def test_all_matlab_assertions(self):
        # Example 1: imag(1i*x) = x.
        f = Ballfun.from_function(lambda x, y, z: 1j * x)
        V = Ballfunv(f, f, f).imag()
        g = Ballfun.from_function(lambda x, y, z: x)
        assert (V - Ballfunv(g, g, g)).norm() < TOL

        # Example 2: imag(x + 1i*y) = y.
        f = Ballfun.from_function(lambda x, y, z: x + 1j * y)
        V = Ballfunv(f, f, f).imag()
        g = Ballfun.from_function(lambda x, y, z: y)
        assert (V - Ballfunv(g, g, g)).norm() < TOL

        # Example 3: mixed components.
        f1 = Ballfun.from_function(lambda x, y, z: x)
        f2 = Ballfun.from_function(lambda x, y, z: 1j * z)
        f3 = Ballfun.from_function(
            lambda x, y, z: jnp.cos(y) + 1j * jnp.sin(x))
        V = Ballfunv(f1, f2, f3).imag()
        g1 = Ballfun.from_function(lambda x, y, z: 0.0 * x)
        g2 = Ballfun.from_function(lambda x, y, z: z)
        g3 = Ballfun.from_function(lambda x, y, z: jnp.sin(x))
        assert (V - Ballfunv(g1, g2, g3)).norm() < TOL
