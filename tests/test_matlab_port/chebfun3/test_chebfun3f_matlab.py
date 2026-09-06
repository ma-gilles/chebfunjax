"""Port of MATLAB Chebfun tests/chebfun3/test_chebfun3f.m (Fable 5).

Each battery member runs in its own process (see ``_battery._sum3_isolated``).

Provenance
----------
MATLAB source : tests/chebfun3/test_chebfun3f.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.chebpref import ChebfunPref

from ._battery import BATTERY, EXACT, _sum3_isolated


class TestChebfun3Chebfun3f:
    @pytest.mark.parametrize("jj", range(len(BATTERY)))
    def test_battery_member(self, jj):
        tol = 1e8 * ChebfunPref().cheb3Prefs.chebfun3eps
        (val,) = _sum3_isolated(jj, False, "chebfun3f")
        assert abs(val - EXACT[jj]) < tol                                   # pass(jj)
