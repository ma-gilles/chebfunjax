"""Port of MATLAB Chebfun tests/ballfun/test_conj.m (Fable 5).

FIXED (Fable 5): Ballfun.conj added in the audit.

Provenance
----------
MATLAB source : tests/ballfun/test_conj.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS

TOL = 1e2 * EPS


class TestBallfunConj:
    def test_all_matlab_assertions(self):
        # conj(x + 1i*y) = x - 1i*y
        f = Ballfun.from_function(lambda x, y, z: x + 1j * y).conj()
        exact = Ballfun.from_function(lambda x, y, z: x - 1j * y)
        assert (f - exact).norm() < TOL
