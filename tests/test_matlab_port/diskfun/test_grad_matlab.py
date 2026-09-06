"""Port of MATLAB Chebfun tests/diskfun/test_grad.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_grad.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.diskfun.diskfunv import Diskfunv

from ._cart import disk_xy

jax.config.update("jax_enable_x64", True)


class TestDiskfunGrad:
    def test_all_matlab_assertions(self):
        tol = 1e2 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = Diskfun.empty().grad()
        assert f.isempty() and isinstance(f, Diskfunv)                       # pass(1)
        assert float(disk_xy(lambda x, y: 0 * x).grad().norm()) < tol        # pass(2)
        u = disk_xy(lambda x, y: x ** 4 * (y - y ** 2)).grad()
        exact = Diskfunv(disk_xy(lambda x, y: 4 * x ** 3 * (y - y ** 2)),
                         disk_xy(lambda x, y: x ** 4 * (1 - 2 * y)))
        assert float((u - exact).norm()) < tol                               # pass(3)
        u = disk_xy(lambda x, y: jnp.cos(4 * x) * jnp.sin(x * y)).grad()
        exact = Diskfunv(disk_xy(lambda x, y: -4 * jnp.sin(4 * x) * jnp.sin(x * y) + jnp.cos(4 * x) * jnp.cos(x * y) * y),
                         disk_xy(lambda x, y: jnp.cos(4 * x) * jnp.cos(x * y) * x))
        assert float((u - exact).norm()) < 2 * tol                           # pass(4)
        g = disk_xy(lambda x, y: jnp.exp(-3 * (x ** 2 + (y + .2) ** 2)))
        u = g.grad()
        exact = Diskfunv(disk_xy(lambda x, y: -6 * x * jnp.exp(-3 * (x ** 2 + (y + .2) ** 2))),
                         disk_xy(lambda x, y: -6 * (y + .2) * jnp.exp(-3 * (x ** 2 + (y + .2) ** 2))))
        assert float((u - exact).norm()) < 10 * tol                          # pass(5)
        gu = g.gradient()
        assert all(a.isequal(b) for a, b in zip(u.components, gu.components))  # pass(6)
