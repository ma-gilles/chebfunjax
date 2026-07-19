"""Port of MATLAB Chebfun tests/chebfun2v/test_roots01.m (Fable 5).

FIXED (Fable 5): Chebfun2v.roots / Chebfun2.roots(f, g) common zeros.

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots01.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun2d import chebfun2

from ._helpers import TOL, match_points, residuals


class TestChebfun2vRoots01:
    def test_all_matlab_assertions(self):
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
        r = f.roots(g)
        assert len(r) == 13
        rf, rg = residuals(f, g, r)
        assert rf < 1e3 * TOL and rg < 1e3 * TOL

        # Linear pair: single intersection [-1/4, 1/4].
        p = chebfun2(lambda x, y: x - y + .5)
        q = chebfun2(lambda x, y: x + y)
        assert match_points(p.roots(q), np.array([[-.25, .25]]), TOL)

        # Linear pair with a known intersection point.
        p = chebfun2(lambda x, y: y + x / 2 + 1 / 10)
        q = chebfun2(lambda x, y: y - 2.1 * x + 2)
        assert match_points(
            p.roots(q),
            np.array([[0.730769230769231, -0.465384615384615]]), TOL)
