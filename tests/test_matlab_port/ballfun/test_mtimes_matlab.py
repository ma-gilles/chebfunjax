"""Port of MATLAB Chebfun tests/ballfun/test_mtimes.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_mtimes.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS, X0, val


class TestBallfunMtimes:
    def test_scalar_mtimes(self):
        f = Ballfun.from_function(lambda x, y, z: x)
        assert abs(val(0.5 * f) - 0.5 * X0) < 1e3 * EPS
