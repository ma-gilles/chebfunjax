"""Port of MATLAB Chebfun tests/ballfun/test_imag.m (Fable 5).

FIXED (Fable 5): Ballfun.imag added in the audit.

Provenance
----------
MATLAB source : tests/ballfun/test_imag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS

TOL = 1e4 * EPS


class TestBallfunImag:
    def test_all_matlab_assertions(self):
        # Example 1: imag(x + 1i*y*z) = y*z
        f = Ballfun.from_function(lambda x, y, z: x + 1j * y * z).imag()
        exact = Ballfun.from_function(lambda x, y, z: y * z)
        assert (f - exact).norm() < TOL

        # Example 2: imag(y) = 0
        f = Ballfun.from_function(lambda x, y, z: y).imag()
        exact = Ballfun.from_function(lambda x, y, z: 0.0 * x)
        assert (f - exact).norm() < TOL
