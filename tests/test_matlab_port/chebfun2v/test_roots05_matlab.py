"""Port of MATLAB Chebfun tests/chebfun2v/test_roots05.m (Fable 5).

FIXED (Fable 5): cross-check of the marching-squares and Bezout-resultant
common-zero finders on a 2*[-1 1 -1 1] rectangle.

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots05.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun2d import chebfun2

from ._helpers import TOL, match_points


class TestChebfun2vRoots05:
    def test_all_matlab_assertions(self):
        rect = (-2.0, 2.0, -2.0, 2.0)
        f = chebfun2(
            lambda x, y: 2 * x * y * jnp.cos(y ** 2) * jnp.cos(2 * x)
            - jnp.cos(x * y), rect)
        g = chebfun2(
            lambda x, y: 2 * jnp.sin(x * y ** 2) * jnp.sin(3 * x * y)
            - jnp.sin(x * y), rect)
        r1 = f.roots(g, method="ms")
        r2 = f.roots(g, method="resultant")
        assert match_points(r1, r2, TOL)
