"""Port of MATLAB Chebfun tests/ballfun/test_sqrt.m (Fable 5).

FIXED (Fable 5): Ballfun.sqrt (compose) exercised by the port.

Provenance
----------
MATLAB source : tests/ballfun/test_sqrt.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS

TOL = 1e8 * EPS


class TestBallfunSqrt:
    def test_all_matlab_assertions(self):
        # Example 1: sqrt(2) constant
        f = Ballfun.from_function(lambda x, y, z: 2.0 + 0.0 * x).sqrt()
        exact = Ballfun.from_function(lambda x, y, z: 2.0 ** 0.5 + 0.0 * x)
        assert (f - exact).norm() < TOL

        # Example 2: sqrt(r^4) = r^2 (spherical)
        f = Ballfun.from_function(
            lambda r, lam, th: r ** 4, spherical=True).sqrt()
        exact = Ballfun.from_function(
            lambda r, lam, th: r ** 2, spherical=True)
        assert (f - exact).norm() < TOL
