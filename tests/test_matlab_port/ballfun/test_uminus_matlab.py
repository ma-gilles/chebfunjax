"""Port of MATLAB Chebfun tests/ballfun/test_uminus.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_uminus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS, X0, val


class TestBallfunUminus:
    def test_negation(self):
        f = Ballfun.from_function(lambda x, y, z: x)
        assert abs(val(-f) + X0) < 1e3 * EPS
