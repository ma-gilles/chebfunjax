"""Port of MATLAB Chebfun tests/ballfun/test_uplus.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_uplus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS


class TestBallfunUplus:
    def test_identity(self):
        f = Ballfun.from_function(lambda x, y, z: x * y + z)
        assert float((f - f).norm()) < 1e3 * EPS
