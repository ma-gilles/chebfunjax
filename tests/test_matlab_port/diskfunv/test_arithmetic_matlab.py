"""Port of MATLAB Chebfun tests/diskfunv/test_arithmetic.m (Fable 5).

MATLAB constructs from cartesian @(x,y) handles; the chebfunjax
Diskfun samples in polar (theta, r), so handles convert via
x = r cos(theta), y = r sin(theta).

Provenance
----------
MATLAB source : tests/diskfunv/test_arithmetic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.diskfun.diskfunv import Diskfunv

jax.config.update("jax_enable_x64", True)

TOL = 1e3 * 2.220446049250313e-16


def _dsk(fc):
    def f(t, r):
        return fc(r * jnp.cos(t), r * jnp.sin(t))
    return Diskfun.from_function(f)


def _norm_inf(g, n=25):
    ts = jnp.linspace(-np.pi + 1e-6, np.pi - 1e-6, n)
    rs = jnp.linspace(1e-3, 1.0 - 1e-6, n)
    T, R = jnp.meshgrid(ts, rs)
    return float(jnp.max(jnp.abs(jnp.asarray(g(T, R)))))


def _vnorm(F, n=25):
    return max(_norm_inf(c, n) for c in F.components)




class TestDiskfunvArithmetic:
    def test_all_matlab_assertions(self):
        f = lambda x, y: jnp.cos(x)
        g = lambda x, y: jnp.sin(y)
        u = Diskfunv(_dsk(f), _dsk(g))
        v = Diskfunv(_dsk(g), _dsk(f))
        plus = Diskfunv(_dsk(lambda x, y: f(x, y) + g(x, y)),
                        _dsk(lambda x, y: g(x, y) + f(x, y)))
        minus = Diskfunv(_dsk(lambda x, y: f(x, y) - g(x, y)),
                         _dsk(lambda x, y: g(x, y) - f(x, y)))
        mult = Diskfunv(_dsk(lambda x, y: f(x, y) * g(x, y)),
                        _dsk(lambda x, y: g(x, y) * f(x, y)))
        assert _vnorm(u + v - plus) < 10 * TOL
        assert _vnorm(u - v - minus) < 10 * TOL
        assert _vnorm(u.times(v) - mult) < 100 * TOL
