"""Port of MATLAB Chebfun tests/ballfun/test_mrdivide.m (Fable 5).

FIXED (Fable 5): Ballfun scalar division exercised by the port.

Provenance
----------
MATLAB source : tests/ballfun/test_mrdivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS

TOL = 1e4 * EPS


class TestBallfunMrdivide:
    def test_all_matlab_assertions(self):
        # Example 1: (x+y)/2
        f = Ballfun.from_function(lambda x, y, z: x + y)
        exact = Ballfun.from_function(lambda x, y, z: (x + y) / 2)
        assert (f / 2 - exact).norm() < TOL

        # Example 2: (x*z)/(-3)
        f = Ballfun.from_function(lambda x, y, z: x * z)
        exact = Ballfun.from_function(lambda x, y, z: -x * z / 3)
        assert (f / (-3) - exact).norm() < TOL

        # Example 3: y/1i = -1i*y
        f = Ballfun.from_function(lambda x, y, z: y)
        exact = Ballfun.from_function(lambda x, y, z: -1j * y)
        assert (f / 1j - exact).norm() < TOL
