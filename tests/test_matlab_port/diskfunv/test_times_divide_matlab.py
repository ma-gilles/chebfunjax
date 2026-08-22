"""Port of MATLAB Chebfun tests/diskfunv/test_times_divide.m (Fable 5).

MATLAB constructs from cartesian @(x,y) handles; the chebfunjax
Diskfun samples in polar (theta, r), so handles convert via
x = r cos(theta), y = r sin(theta).

Provenance
----------
MATLAB source : tests/diskfunv/test_times_divide.m
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




class TestDiskfunvTimesDivide:
    def test_all_matlab_assertions(self):
        f = _dsk(lambda x, y: jnp.sin(y) + jnp.cos(2 * x * y))
        F = Diskfunv(f, f)
        G = Diskfunv(2 * f, 2 * f)
        H = Diskfunv(f * 0.5, f * 0.5)
        assert _vnorm(2 * F - G) < TOL
        assert _vnorm(F * 2 - G) < TOL
        assert _vnorm(F * 0.5 - H) < TOL
