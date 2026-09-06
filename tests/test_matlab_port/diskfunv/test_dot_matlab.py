"""Port of MATLAB Chebfun tests/diskfunv/test_dot.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfunv/test_dot.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2, chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun.diskfunv import Diskfunv

from ..diskfun._cart import disk_xy

jax.config.update("jax_enable_x64", True)


class TestDiskfunvDot:
    def test_all_matlab_assertions(self):
        tol = 10 * ChebfunPref().cheb2Prefs.chebfun2eps
        h = Diskfunv.empty().dot(Diskfunv.empty())
        assert h.isempty()                                                   # pass(1)
        F = Chebfun2v([chebfun2(lambda x, y: jnp.cos(x)).approx, chebfun2(lambda x, y: jnp.sin(y)).approx])
        G = Chebfun2v([chebfun2(lambda x, y: x).approx, chebfun2(lambda x, y: y).approx])
        dotF1 = Chebfun2(approx=F.dot(G))
        dotF2 = Chebfun2(approx=F.T * G)
        assert float((dotF1 - dotF2).norm()) < tol                           # pass(2)
        u1 = disk_xy(lambda x, y: x * jnp.cos(2 * y))
        u2 = disk_xy(lambda x, y: y * jnp.sin(2 * x))
        u = Diskfunv(u1, u2)
        v1 = disk_xy(lambda x, y: x * y)
        v2 = disk_xy(lambda x, y: y ** 2)
        v = Diskfunv(v1, v2)
        f = u.dot(v)
        g = u1 * v1 + u2 * v2
        assert float((g - f).norm()) < tol                                   # pass(3)
        g = u.T * v
        assert float((g - f).norm()) < tol                                   # pass(4)
