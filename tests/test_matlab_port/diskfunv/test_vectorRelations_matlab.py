"""Port of MATLAB Chebfun tests/diskfunv/test_vectorRelations.m (Fable 5).

MATLAB constructs from cartesian @(x,y) handles; the chebfunjax
Diskfun samples in polar (theta, r), so handles convert via
x = r cos(theta), y = r sin(theta).

Provenance
----------
MATLAB source : tests/diskfunv/test_vectorRelations.m
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


class TestDiskfunvVectorrelations:
    def test_all_matlab_assertions(self):
        f = _dsk(lambda x, y: jnp.cos((y + 0.1) * x))
        g = f.gradient()
        # pass(1): div(grad f) = laplacian f.
        dg = (g.components[0].diffx() + g.components[1].diffy())
        assert _norm_inf(dg - f.laplacian()) < 3e7 * TOL
        # pass(2): div(curl f) = 0.
        c = f.curl_scalar()
        dc = c.components[0].diffx() + c.components[1].diffy()
        assert _norm_inf(dc) < 3e7 * TOL
        # pass(3): curl(grad f) = 0 (scalar curl of a 2-D field).
        cg = (g.components[1].diffx() - g.components[0].diffy())
        assert _norm_inf(cg) < 3e7 * TOL
        # pass(4): divgrad identity.
        f2 = _dsk(lambda x, y: jnp.cos(y + x) * jnp.sin(np.pi * x))
        g2 = _dsk(lambda x, y: jnp.sin(np.pi * x * y))
        F = Diskfunv(f2, g2)
        ident = (f2.diffx().diffx() + g2.diffy().diffy())
        assert _norm_inf(F.divgrad() - ident) < 3e7 * TOL
