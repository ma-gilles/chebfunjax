"""Port of MATLAB Chebfun tests/diskfun/test_curl.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_curl.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun.diskfunv import Diskfunv

from ._cart import disk_xy

jax.config.update("jax_enable_x64", True)


class TestDiskfunCurl:
    def test_all_matlab_assertions(self):
        tol = 1e2 * ChebfunPref().cheb2Prefs.chebfun2eps
        u = disk_xy(lambda x, y: x * y).curl()
        assert isinstance(u, Diskfunv)                                       # pass(1)
        assert float(disk_xy(lambda x, y: 0 * x).curl().norm()) < tol        # pass(2)
        u = disk_xy(lambda x, y: x ** 2 - y ** 3).curl()
        exact = Diskfunv(disk_xy(lambda x, y: -3 * y ** 2), disk_xy(lambda x, y: -2 * x))
        assert float((u - exact).norm()) < tol                               # pass(3)
        u = disk_xy(lambda x, y: jnp.cos(4 * x)).curl()
        exact = Diskfunv(disk_xy(lambda x, y: 0 * x), disk_xy(lambda x, y: 4 * jnp.sin(4 * x)))
        assert float((u - exact).norm()) < 10 * tol                          # pass(4)
        u = disk_xy(lambda x, y: jnp.cos(4 * x ** 2) * jnp.sin(y)).curl()
        exact = Diskfunv(disk_xy(lambda x, y: jnp.cos(4 * x ** 2) * jnp.cos(y)),
                         disk_xy(lambda x, y: 8 * x * jnp.sin(4 * x ** 2) * jnp.sin(y)))
        assert float((u - exact).norm()) < 30 * tol                          # pass(5)
