"""Port of MATLAB Chebfun tests/chebfun3/test_battery.m (Fable 5).

Each battery member runs in its own process (see ``_battery._sum3_isolated``).

Provenance
----------
MATLAB source : tests/chebfun3/test_battery.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
import pytest

from chebfunjax.chebpref import ChebfunPref

from ._battery import BATTERY, EXACT, _sum3_isolated


class TestChebfun3Battery:
    @pytest.mark.parametrize("jj", range(len(BATTERY)))
    def test_battery_member(self, jj):
        tol = 1e8 * ChebfunPref().cheb3Prefs.chebfun3eps
        vals = _sum3_isolated(jj, True, "battery")
        assert np.max(np.abs(np.asarray(vals) - EXACT[jj])) < tol         # pass(5jj-4 .. 5jj)
