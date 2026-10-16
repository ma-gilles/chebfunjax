"""Port of MATLAB Chebfun tests/chebfun2v/test_roots_slow.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots_slow.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun2d import chebfun2

from ._helpers import TOL, match_points

# This original stress case runs serially in CPU qualification.


class TestChebfun2vRootsSlow:
    def test_all_matlab_assertions(self):
        f = chebfun2(
            lambda x, y: jnp.exp(x - 2 * x ** 2 - y ** 2)
            * jnp.sin(10 * (x + y + x * y ** 2)))
        g = chebfun2(
            lambda x, y: jnp.exp(-x + 2 * y ** 2 + x * y ** 2)
            * jnp.sin(10 * (x - y - 2 * x * y ** 2)))
        r1 = f.roots(g, method="ms")
        r2 = f.roots(g, method="resultant")
        assert match_points(r1, r2, TOL)
