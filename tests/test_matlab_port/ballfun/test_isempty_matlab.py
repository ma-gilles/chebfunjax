"""Port of MATLAB Chebfun tests/ballfun/test_isempty.m (Fable 5).

FIXED: Ballfun.empty()/isempty() added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/ballfun/test_isempty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun


class TestBallfunIsempty:
    def test_empty_and_nonempty(self):
        assert Ballfun.empty().isempty()
        f = Ballfun.from_function(lambda x, y, z: x)
        assert not f.isempty()
