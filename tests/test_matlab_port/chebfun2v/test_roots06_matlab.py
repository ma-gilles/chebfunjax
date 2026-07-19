"""Port of MATLAB Chebfun tests/chebfun2v/test_roots06.m (Fable 5).

FIXED (Fable 5): Chebfun2v common zeros (residual + known roots).

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots06.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun2d import chebfun2

from ._helpers import TOL, match_points, residuals


class TestChebfun2vRoots06:
    def test_all_matlab_assertions(self):
        # Residual check.
        f = chebfun2(lambda x, y: (y - 2 * x) * (y + 0.5 * x))
        g = chebfun2(lambda x, y: x * (x ** 2 + y ** 2 - 1))
        r = f.roots(g)
        rf, rg = residuals(f, g, r)
        assert rf < TOL and rg < TOL

        # Six known common zeros.
        s = np.sqrt(5.0)
        f = chebfun2(lambda x, y: (y - 2 * x) * (y + .5 * x))
        g = chebfun2(lambda x, y: (x - .0001) * (x ** 2 + y ** 2 - 1))
        exact = np.array([
            [1 / 10000, -1 / 20000],
            [1 / 10000, 1 / 5000],
            [-2 / s, 1 / s],
            [-1 / s, -2 / s],
            [1 / s, 2 / s],
            [2 / s, -1 / s],
        ])
        assert match_points(f.roots(g), exact, 10 * TOL)
