"""Port of MATLAB Chebfun tests/chebfun2v/test_roots01.m (Fable 5).

FIXED (Fable 5): Chebfun2v.roots / Chebfun2.roots(f, g) common zeros.

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots01.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d import chebfun2

from ._helpers import TOL, match_points


class TestChebfun2vRoots01:
    def test_original_product_thirteen_roots(self):
        # Big product pair with 13 common zeros.
        f = chebfun2(
            lambda x, y: (y ** 2 - x ** 3) * ((y - 0.7) ** 2 - (x - 0.3) ** 3)
            * ((y + 0.2) ** 2 - (x + 0.8) ** 3)
            * ((y + 0.2) ** 2 - (x - 0.8) ** 3))
        g = chebfun2(
            lambda x, y: ((y + .4) ** 3 - (x - .4) ** 2)
            * ((y + .3) ** 3 - (x - .3) ** 2)
            * ((y - .5) ** 3 - (x + .6) ** 2)
            * ((y + 0.3) ** 3 - (2 * x - 0.8) ** 3))
        r = jnp.asarray(f.roots(g, method="resultant"))
        assert len(r) == 13
        assert float(jnp.linalg.norm(f(r[:, 0], r[:, 1]))) < 1e3 * TOL
        assert float(jnp.linalg.norm(g(r[:, 0], r[:, 1]))) < 1e3 * TOL

    def test_original_default_linear(self):
        # Linear pair: single intersection [-1/4, 1/4].
        p = chebfun2(lambda x, y: x - y + .5)
        q = chebfun2(lambda x, y: x + y)
        assert match_points(p.roots(q), np.array([[-.25, .25]]), TOL)

    def test_original_explicit_linear(self):
        # Linear pair with a known intersection point.
        p = chebfun2(lambda x, y: y + x / 2 + 1 / 10)
        q = chebfun2(lambda x, y: y - 2.1 * x + 2)
        assert match_points(
            p.roots(q, method="resultant"),
            np.array([[0.730769230769231, -0.465384615384615]]), TOL)

    def test_original_sextic_comparison(self):
        f = chebfun2(lambda x, y: 144*(x**4+y**4)-225*(x**2+y**2)+350*x**2*y**2+81)
        g = chebfun2(lambda x, y: y-x**6)
        assert match_points(f.roots(g, method="ms"), f.roots(g, method="resultant"), TOL)

    def test_original_cusp_comparison(self):
        f = chebfun2(lambda x, y: y**2-x**3)
        g = chebfun2(lambda x, y: (y+.1)**3-(x-.1)**2)
        assert match_points(f.roots(g, method="ms"), f.roots(g, method="resultant"), TOL)
