"""Port of MATLAB Chebfun tests/ballfunv/test_isempty.m (Fable 5).

FIXED: Ballfunv.empty()/isempty() added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/ballfunv/test_isempty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv


class TestBallfunvEmpty:
    def test_empty_and_nonempty(self):
        assert Ballfunv.empty().isempty()
        v = Ballfunv(Ballfun.from_function(lambda x, y, z: x), Ballfun.from_function(lambda x, y, z: y), Ballfun.from_function(lambda x, y, z: z))
        assert not v.isempty()
