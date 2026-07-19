"""Port of MATLAB Chebfun tests/ballfunv/test_iszero.m (Fable 5).

FIXED (Fable 5): Ballfunv.iszero added in the audit (all components).

Provenance
----------
MATLAB source : tests/ballfunv/test_iszero.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv


class TestBallfunvIszero:
    def test_all_matlab_assertions(self):
        f = Ballfun.from_function(lambda x, y, z: 1.0 + 0.0 * x)
        F = Ballfunv(f, f, f)
        assert (F - F).iszero()
