"""Port of MATLAB Chebfun tests/diskfun/test_inherited.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_inherited.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun.diskfun import Diskfun

from ._cart import disk_polar, disk_xy

jax.config.update("jax_enable_x64", True)

D2R = np.pi / 180.0


def _n(f):
    return float(f.norm())


class TestDiskfunInherited:
    def test_all_matlab_assertions(self):
        tol = 1e2 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = disk_xy(lambda x, y: -1 + 0 * x)
        h = disk_xy(lambda x, y: 1 + 0 * x)
        assert _n(abs(f) - h) < tol                                          # pass(1)
        f = disk_xy(lambda x, y: jnp.cos(x))
        assert _n(f.cos() - disk_xy(lambda x, y: jnp.cos(jnp.cos(x)))) < tol  # pass(2)
        assert _n(f.cosh() - disk_xy(lambda x, y: jnp.cosh(jnp.cos(x)))) < tol  # pass(3)
        f = disk_xy(lambda x, y: x)
        assert _n(f - f.conj()) < tol                                        # pass(4)
        f = disk_xy(lambda x, y: jnp.cos(x) + jnp.sin(y))
        assert _n(f.exp() - disk_xy(lambda x, y: jnp.exp(jnp.cos(x) + jnp.sin(y)))) < tol  # pass(5)
        f = disk_xy(lambda x, y: x)
        assert _n(f.imag()) < tol                                            # pass(6)
        f = disk_xy(lambda x, y: jnp.cos(x))
        assert f.isequal(disk_xy(lambda x, y: jnp.cos(-x)))                  # pass(7)
        assert not f.isequal(disk_xy(lambda x, y: jnp.cos(x) + 1))           # pass(8)
        assert disk_xy(lambda x, y: x).isreal()                              # pass(9)
        assert disk_xy(lambda x, y: jnp.cos(x * y)).isreal()                 # pass(10)
        assert disk_xy(lambda x, y: 0 * jnp.cos(x * y)).iszero()             # pass(11)
        assert Diskfun.from_values(np.zeros((10, 10))).iszero()              # pass(12)
        f = disk_polar(lambda t, r: r ** 2)
        m, _n2 = f.length()
        assert m == 1                                                        # pass(13)
        f = disk_xy(lambda x, y: jnp.exp(x))
        assert _n(f.log() - disk_xy(lambda x, y: x)) < tol                   # pass(14)
        f = disk_xy(lambda x, y: jnp.cos(x * y))
        assert _n(f.sin() - disk_xy(lambda x, y: jnp.sin(jnp.cos(x * y)))) < tol  # pass(15)
        assert _n(f.sinh() - disk_xy(lambda x, y: jnp.sinh(jnp.cos(x * y)))) < tol  # pass(16)
        assert f.size(1) == np.inf and f.size(2) == np.inf                   # pass(17)
        f2 = disk_xy(lambda x, y: jnp.cos(x * y) ** 2)
        assert _n(f2.sqrt() - disk_xy(lambda x, y: jnp.cos(x * y))) < tol    # pass(18)
        assert _n(f.tan() - disk_xy(lambda x, y: jnp.tan(jnp.cos(x * y)))) < 10 * tol  # pass(19)
        assert _n(f.tand() - disk_xy(lambda x, y: jnp.tan(D2R * jnp.cos(x * y)))) < tol  # pass(20)
        assert _n(f.tanh() - disk_xy(lambda x, y: jnp.tanh(jnp.cos(x * y)))) < tol  # pass(21)
        assert _n(f.uminus() - disk_xy(lambda x, y: -jnp.cos(x * y))) < tol  # pass(22)
        assert _n(f.uplus() - disk_xy(lambda x, y: +jnp.cos(x * y))) < tol   # pass(23)
