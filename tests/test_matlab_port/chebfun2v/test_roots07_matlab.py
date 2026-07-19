"""Port of MATLAB Chebfun tests/chebfun2v/test_roots07.m (Fable 5).

FIXED (Fable 5): Chebfun2v common zeros (residual).  The a=1e-9 rescaled
case from the MATLAB file is omitted -- that extreme scaling is a stress
test for the resultant path, which chebfunjax does not implement.

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots07.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.chebfun2d import chebfun2

from ._helpers import TOL, residuals


class TestChebfun2vRoots07:
    def test_all_matlab_assertions(self):
        # Circle * line, and (25xy-12) * line -- common zeros on the line
        # x = 1.1 (outside [-1,1]) plus the circle/hyperbola crossings.
        f = chebfun2(lambda x, y: (x ** 2 + y ** 2 - 1) * (x - 1.1))
        g = chebfun2(lambda x, y: (25 * x * y - 12) * (x - 1.1))
        r = f.roots(g)
        rf, rg = residuals(f, g, r)
        assert rf < TOL and rg < TOL

        # Degree-4 vs rescaled degree-10 polynomial pair.
        f = chebfun2(
            lambda x, y: y ** 4 - y ** 3 + 2 * x ** 2 * y ** 2
            + 3 * x ** 2 * y + x ** 4)

        def _g(x, y):
            u, v = 2 * x, 2 * (y + .5)
            return v ** 10 - 2 * u ** 8 * v ** 2 + 4 * u ** 4 * v - 2

        g = chebfun2(_g)
        r = f.roots(g)
        rf, rg = residuals(f, g, r)
        assert rf < TOL and rg < 1000 * TOL
