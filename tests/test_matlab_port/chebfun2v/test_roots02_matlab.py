"""Port of MATLAB Chebfun tests/chebfun2v/test_roots02.m (Fable 5).

FIXED (Fable 5): the marching-squares and Bezout-resultant common-zero
finders are both implemented, so the MATLAB cross-check is ported directly
(``roots(F, 'ms')`` and ``roots(F, 'resultant')`` must agree), with an
additional residual check on the found common zeros.

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots02.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d import chebfun2

from ._helpers import TOL, match_points, residuals


class TestChebfun2vRoots02:
    def test_all_matlab_assertions(self):
        cases = [
            (lambda x, y: jnp.cos(10 * x * y), lambda x, y: x + y ** 2),
            (lambda x, y: x, lambda x, y: (x - .9999) ** 2 + y ** 2 - 1),
            (lambda x, y: jnp.sin(4 * (x + y / 10 + np.pi / 10)),
             lambda x, y: jnp.cos(2 * (x - 2 * y + np.pi / 7))),
        ]
        for ff, gg in cases:
            f = chebfun2(ff)
            g = chebfun2(gg)
            r1 = f.roots(g, method="ms")
            r2 = f.roots(g, method="resultant")
            assert len(r1) > 0
            assert match_points(r1, r2, TOL)
            rf, rg = residuals(f, g, r2)
            assert rf < 1e3 * TOL and rg < 1e3 * TOL
