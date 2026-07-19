"""Port of MATLAB Chebfun tests/spherefun/test_iszero.m (Fable 5).

FIXED (Fable 5): Spherefun.iszero added in the audit.

Provenance
----------
MATLAB source : tests/spherefun/test_iszero.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.spherefun.spherefun import Spherefun


class TestSpherefunIszero:
    def test_all_matlab_assertions(self):
        # The zero function.
        f = Spherefun.from_function(lambda lam, th: 0.0 * lam)
        assert f.iszero()

        # cos(x) on the sphere (x = cos(lam) sin(th)) is nonzero.
        f = Spherefun.from_function(
            lambda lam, th: jnp.cos(jnp.cos(lam) * jnp.sin(th)))
        assert not f.iszero()
