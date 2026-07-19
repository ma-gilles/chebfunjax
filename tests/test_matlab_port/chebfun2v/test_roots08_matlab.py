"""Port of MATLAB Chebfun tests/chebfun2v/test_roots08.m (Fable 5).

FIXED (Fable 5): cross-check of the marching-squares and Bezout-resultant
common-zero finders.

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots08.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun2d import chebfun2

from ._helpers import TOL, match_points


class TestChebfun2vRoots08:
    def test_all_matlab_assertions(self):
        f = chebfun2(lambda x, y: jnp.sin(10 * x - y / 10))
        g = chebfun2(lambda x, y: jnp.cos(3 * x * y))
        r1 = f.roots(g, method="ms")
        r2 = f.roots(g, method="resultant")
        assert match_points(r1, r2, TOL)

        f = chebfun2(lambda x, y: jnp.sin(10 * x - y / 10) + y)
        g = chebfun2(lambda x, y: jnp.cos(10 * y - x / 10) - x)
        r1 = f.roots(g, method="ms")
        r2 = f.roots(g, method="resultant")
        assert match_points(r1, r2, TOL)
