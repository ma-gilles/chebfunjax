"""Port of MATLAB Chebfun tests/diskfunv/test_transpose.m (Fable 5).

MATLAB constructs from cartesian @(x,y) handles; the chebfunjax
Diskfun samples in polar (theta, r), so handles convert via
x = r cos(theta), y = r sin(theta).

Provenance
----------
MATLAB source : tests/diskfunv/test_transpose.m
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




class TestDiskfunvTranspose:
    def test_all_matlab_assertions(self):
        e = Diskfunv.empty()
        assert e.transpose().isempty()  # pass(1)/(2)
        assert e.ctranspose().isempty()
        f = _dsk(lambda x, y: jnp.cos((x + 0.1) * y))
        u = f.gradient()
        m, n, p = u.ctranspose().size
        assert np.isinf(m) and np.isinf(n) and p == 2  # pass(3)
        m, n, p = u.transpose().size
        assert np.isinf(m) and np.isinf(n) and p == 2  # pass(4)
        rng = np.random.RandomState(10)
        t0 = jnp.asarray(float(rng.rand()))
        r0 = jnp.asarray(float(rng.rand()))
        for a, b in zip(u.ctranspose().components,
                        u.transpose().components):
            assert abs(float(a(t0, r0)) - float(b(t0, r0))) < TOL
        v = u.transpose()
        for a, b in zip(u.components, v.transpose().components):
            assert _norm_inf(a - b) < TOL  # pass(6)
        s = u.ctranspose() @ u  # pass(7)
        assert isinstance(s, Diskfun)
