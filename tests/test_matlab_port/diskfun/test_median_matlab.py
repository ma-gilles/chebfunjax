"""Port of MATLAB Chebfun tests/diskfun/test_median.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_median.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun.diskfun import Diskfun

from ._cart import disk_xy

jax.config.update("jax_enable_x64", True)


class TestDiskfunMedian:
    def test_all_matlab_assertions(self):
        tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
        assert Diskfun.empty().median() is None                              # pass(1)
        g = disk_xy(lambda x, y: 0 * x + 1)
        assert float((g.median() - 1).norm(2)) < tol                         # pass(2)
