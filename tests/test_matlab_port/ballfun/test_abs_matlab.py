"""Port of MATLAB Chebfun tests/ballfun/test_abs.m (Fable 5).

FIXED (Fable 5): Ballfun.abs added in the audit.

Provenance
----------
MATLAB source : tests/ballfun/test_abs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS

TOL = 1e2 * EPS


class TestBallfunAbs:
    def test_all_matlab_assertions(self):
        # Example 1: |1i*x^2| = x^2
        f = abs(Ballfun.from_function(lambda x, y, z: 1j * x ** 2))
        exact = Ballfun.from_function(lambda x, y, z: x ** 2)
        assert (f - exact).norm() < TOL

        # Example 2: |(x+1i*y)^2| = x^2 + y^2
        f = abs(Ballfun.from_function(lambda x, y, z: (x + 1j * y) ** 2))
        exact = Ballfun.from_function(lambda x, y, z: x ** 2 + y ** 2)
        assert (f - exact).norm() < TOL

        # Example 3: |r^2| = r^2 (spherical)
        exact = Ballfun.from_function(
            lambda r, lam, th: r ** 2, spherical=True)
        f = abs(exact)
        assert (f - exact).norm() < TOL

        # Example 4: |x^2| = x^2
        exact = Ballfun.from_function(lambda x, y, z: x ** 2)
        f = abs(exact)
        assert (f - exact).norm() < TOL
