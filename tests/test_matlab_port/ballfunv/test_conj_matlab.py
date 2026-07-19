"""Port of MATLAB Chebfun tests/ballfunv/test_conj.m (Fable 5).

FIXED (Fable 5): Ballfunv.conj added in the audit (componentwise).

Provenance
----------
MATLAB source : tests/ballfunv/test_conj.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv

from ._helpers import EPS

TOL = 1e2 * EPS


class TestBallfunvConj:
    def test_all_matlab_assertions(self):
        # Example 1: conj of a real field is itself.
        f = Ballfun.from_function(lambda x, y, z: z)
        V = Ballfunv(f, f, f).conj()
        g = Ballfun.from_function(lambda x, y, z: z)
        assert (V - Ballfunv(g, g, g)).norm() < TOL

        # Example 2: conj(y + 1i*z) = y - 1i*z.
        f = Ballfun.from_function(lambda x, y, z: y + 1j * z)
        V = Ballfunv(f, f, f).conj()
        g = Ballfun.from_function(lambda x, y, z: y - 1j * z)
        assert (V - Ballfunv(g, g, g)).norm() < TOL

        # Example 3: mixed components.
        f1 = Ballfun.from_function(lambda x, y, z: x)
        f2 = Ballfun.from_function(lambda x, y, z: 1j * z)
        f3 = Ballfun.from_function(
            lambda x, y, z: jnp.cos(y) + 1j * jnp.sin(x))
        V = Ballfunv(f1, f2, f3).conj()
        g1 = Ballfun.from_function(lambda x, y, z: x)
        g2 = Ballfun.from_function(lambda x, y, z: -1j * z)
        g3 = Ballfun.from_function(
            lambda x, y, z: jnp.cos(y) - 1j * jnp.sin(x))
        assert (V - Ballfunv(g1, g2, g3)).norm() < TOL
