"""Port of MATLAB Chebfun tests/ballfun/test_minus.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_minus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS, X0, Y0, val


class TestBallfunMinus:
    def test_minus_and_self_cancellation(self):
        f = Ballfun.from_function(lambda x, y, z: x)
        g = Ballfun.from_function(lambda x, y, z: y * y)
        assert abs(val(f - g) - (X0 - Y0 ** 2)) < 1e3 * EPS
        assert float((f - f).norm()) < 1e3 * EPS
