"""Port of MATLAB Chebfun tests/chebfun2v/test_roots03.m (Fable 5).

FIXED (Fable 5): Chebfun2v common zeros (count + residual).

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots03.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun2d import chebfun2

from ._helpers import TOL


class TestChebfun2vRoots03:
    def test_original_four_roots(self):
        # Ellipse intersecting a product of three circles: 4 common zeros.
        f = chebfun2(lambda x, y: (x - .3) ** 2 + 2 * (y + 0.3) ** 2 - 1)
        g = chebfun2(
            lambda x, y: ((x - .49) ** 2 + (y + .5) ** 2 - 1)
            * ((x + 0.5) ** 2 + (y + 0.5) ** 2 - 1)
            * ((x - 1) ** 2 + (y - 0.5) ** 2 - 1))
        r = jnp.asarray(f.roots(g, method="resultant"))
        assert len(r) == 4
        assert float(jnp.linalg.norm(f(r[:, 0], r[:, 1]))) < TOL
        assert float(jnp.linalg.norm(g(r[:, 0], r[:, 1]))) < 1e2 * TOL

    def test_original_fortyfive_roots(self):
        # Product of four ellipses vs product of four ellipses: 45 zeros.
        f = chebfun2(
            lambda x, y: ((x - 0.1) ** 2 + 2 * (y - 0.1) ** 2 - 1)
            * ((x + 0.3) ** 2 + 2 * (y - 0.2) ** 2 - 1)
            * ((x - 0.3) ** 2 + 2 * (y + 0.15) ** 2 - 1)
            * ((x - 0.13) ** 2 + 2 * (y + 0.15) ** 2 - 1))
        g = chebfun2(
            lambda x, y: (2 * (x + 0.1) ** 2 + (y + 0.1) ** 2 - 1)
            * (2 * (x + 0.1) ** 2 + (y - 0.1) ** 2 - 1)
            * (2 * (x - 0.3) ** 2 + (y - 0.15) ** 2 - 1)
            * ((x - 0.21) ** 2 + 2 * (y - 0.15) ** 2 - 1))
        r = jnp.asarray(f.roots(g, method="resultant"))
        assert len(r) == 45
        assert float(jnp.linalg.norm(f(r[:, 0], r[:, 1]))) < 10 * TOL
        assert float(jnp.linalg.norm(g(r[:, 0], r[:, 1]))) < 100 * TOL
