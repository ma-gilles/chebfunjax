"""Port of MATLAB Chebfun tests/diskfunv/test_diff.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfunv/test_diff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun.diskfunv import Diskfunv

from ..diskfun._cart import disk_xy

jax.config.update("jax_enable_x64", True)

pi = np.pi


def _n(F):
    return float(F.norm())


class TestDiskfunvDiff:
    def test_all_matlab_assertions(self):
        tol = 1e3 * ChebfunPref().cheb2Prefs.chebfun2eps
        u = Diskfunv.empty()
        for f in (u.diffx(), u.diffy(), u.diff(2, 2)):
            assert f.isempty() and isinstance(f, Diskfunv)                   # pass(1)-(3)
        f = disk_xy(lambda x, y: jnp.cos(3 * y ** 2) * jnp.sin(2 * x))
        g = disk_xy(lambda x, y: jnp.sin(x ** 4) * jnp.cos(5 * y))
        F = Diskfunv(f, g)
        G = Diskfunv(f.diff(1, 1), g.diff(1, 1))
        assert _n(2 * F.diffx() - F.diff(1, 1) - G) < tol                    # pass(4)
        G = Diskfunv(f.diff(1, 3), g.diff(1, 3))
        assert _n(2 * F.diffx(3) - F.diff(1, 3) - G) < tol                   # pass(5)
        G = Diskfunv(f.diff(2, 1), g.diff(2, 1))
        assert _n(2 * F.diffy() - F.diff(2, 1) - G) < tol                    # pass(6)
        G = Diskfunv(f.diff(2, 2), g.diff(2, 2))
        assert _n(2 * F.diffy(2) - F.diff(2, 2) - G) < tol                   # pass(7)
        F = Diskfunv(disk_xy(lambda x, y: jnp.cos(x)), disk_xy(lambda x, y: jnp.sin(y)))
        exact = Diskfunv(disk_xy(lambda x, y: -jnp.sin(x)), disk_xy(lambda x, y: 0 * x))
        assert _n(F.diffx() - exact) < tol                                   # pass(8)
        exact = Diskfunv(disk_xy(lambda x, y: 0 * x), disk_xy(lambda x, y: jnp.cos(y)))
        assert _n(F.diffy() - exact) < tol                                   # pass(9)
        f = disk_xy(lambda x, y: jnp.cos(3 * y ** 2) * jnp.sin(2 * x))
        g = disk_xy(lambda x, y: jnp.sin(pi * x ** 4) * jnp.cos(5 * y))
        F = Diskfunv(f, g)
        exact = Diskfunv(disk_xy(lambda x, y: 2 * jnp.cos(3 * y ** 2) * jnp.cos(2 * x)),
                         disk_xy(lambda x, y: 4 * pi * x ** 3 * jnp.cos(pi * x ** 4) * jnp.cos(5 * y)))
        assert _n(F.diffx() - exact) < tol                                   # pass(10)
        exact = Diskfunv(disk_xy(lambda x, y: -6 * y * jnp.sin(3 * y ** 2) * jnp.sin(2 * x)),
                         disk_xy(lambda x, y: -5 * jnp.sin(5 * y) * jnp.sin(pi * x ** 4)))
        assert _n(F.diffy() - exact) < tol                                   # pass(11)
