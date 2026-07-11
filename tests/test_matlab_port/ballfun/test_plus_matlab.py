"""Port of MATLAB Chebfun tests/ballfun/test_plus.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_plus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS, X0, Y0, val


class TestBallfunPlus:
    def test_plus_functions_and_scalar(self):
        f = Ballfun.from_function(lambda x, y, z: x)
        g = Ballfun.from_function(lambda x, y, z: y * y)
        assert abs(val(f + g) - (X0 + Y0 ** 2)) < 1e3 * EPS
        assert abs(val(f + 2.0) - (X0 + 2)) < 1e3 * EPS
        assert abs(val(2.0 + f) - (X0 + 2)) < 1e3 * EPS
