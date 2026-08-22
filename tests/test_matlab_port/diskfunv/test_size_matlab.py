"""Port of MATLAB Chebfun tests/diskfunv/test_size.m (Fable 5).

MATLAB constructs from cartesian @(x,y) handles; the chebfunjax
Diskfun samples in polar (theta, r), so handles convert via
x = r cos(theta), y = r sin(theta).

Provenance
----------
MATLAB source : tests/diskfunv/test_size.m
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


class TestDiskfunvSize:
    def test_all_matlab_assertions(self):
        F = Diskfunv(_dsk(lambda x, y: jnp.cos(x)),
                     _dsk(lambda x, y: jnp.sin(y)))
        assert F.size == (2, np.inf, np.inf)  # pass(1)-(4)
        G = F.ctranspose()
        assert G.size == (np.inf, np.inf, 2)  # pass(5)-(8)
