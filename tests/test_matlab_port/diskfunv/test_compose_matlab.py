"""Port of MATLAB Chebfun tests/diskfunv/test_compose.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfunv/test_compose.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun.diskfunv import Diskfunv

from ..diskfun._cart import disk_xy

jax.config.update("jax_enable_x64", True)


class TestDiskfunvCompose:
    def test_all_matlab_assertions(self):
        tol = 1e3 * ChebfunPref().cheb2Prefs.chebfun2eps
        F = Diskfunv(disk_xy(lambda x, y: x), disk_xy(lambda x, y: y))
        g = chebfun2(lambda x, y: x + y)
        h = F.compose(g)
        h_true = disk_xy(lambda x, y: x + y)
        assert float((h - h_true).norm()) < tol                              # pass(1)
        G = Chebfun2v([chebfun2(lambda x, y: x + y).approx, chebfun2(lambda x, y: x - y).approx])
        H = F.compose(G)
        H_true = Diskfunv(disk_xy(lambda x, y: x + y), disk_xy(lambda x, y: x - y))
        assert float((H - H_true).norm()) < tol                              # pass(2)
