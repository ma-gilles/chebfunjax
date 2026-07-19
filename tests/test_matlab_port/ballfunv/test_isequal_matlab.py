"""Port of MATLAB Chebfun tests/ballfunv/test_isequal.m (Fable 5).

FIXED (Fable 5): Ballfunv.isequal added in the audit.

Provenance
----------
MATLAB source : tests/ballfunv/test_isequal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv


class TestBallfunvIsequal:
    def test_all_matlab_assertions(self):
        f = Ballfun.from_function(lambda x, y, z: 1.0 + 0.0 * x)
        F = Ballfunv(f, f, f)
        G = F + F - F
        assert F.isequal(G) and G.isequal(F)
