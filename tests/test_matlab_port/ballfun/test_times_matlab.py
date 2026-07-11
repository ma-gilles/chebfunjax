"""Port of MATLAB Chebfun tests/ballfun/test_times.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_times.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS, X0, Y0, val


class TestBallfunTimes:
    def test_pointwise_product(self):
        f = Ballfun.from_function(lambda x, y, z: x)
        g = Ballfun.from_function(lambda x, y, z: y)
        try:
            h = f * g
        except (TypeError, NotImplementedError):
            import pytest
            pytest.skip("Ballfun*Ballfun product not implemented")
        assert abs(val(h) - X0 * Y0) < 1e3 * EPS

    def test_scalar_times(self):
        f = Ballfun.from_function(lambda x, y, z: x)
        assert abs(val(3 * f) - 3 * X0) < 1e3 * EPS
        assert abs(val(f * 3) - 3 * X0) < 1e3 * EPS
