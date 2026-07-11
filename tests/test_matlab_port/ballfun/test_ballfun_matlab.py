"""Port of MATLAB Chebfun tests/ballfun/test_ballfun.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_ballfun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS, X0, Y0, Z0, val


class TestBallfunBallfun:
    def test_construction_and_eval(self):
        f = Ballfun.from_function(lambda x, y, z: 1 + x * y + z ** 2)
        exact = 1 + X0 * Y0 + Z0 ** 2
        assert abs(val(f) - exact) < 1e3 * EPS
