"""Port of MATLAB Chebfun tests/diskfunv/test_jacobian.m (Fable 5).

MATLAB constructs from cartesian @(x,y) handles; the chebfunjax
Diskfun samples in polar (theta, r), so handles convert via
x = r cos(theta), y = r sin(theta).

Provenance
----------
MATLAB source : tests/diskfunv/test_jacobian.m
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


class TestDiskfunvJacobian:
    def test_all_matlab_assertions(self):
        # pass(1): empty in, empty diskfun out.
        assert Diskfunv.empty().jacobian().isempty()
        # pass(2)/(3): Jacobian determinant identity + exact value.
        F = Diskfunv(
            _dsk(lambda x, y: jnp.cos(np.pi * x * y)),
            _dsk(lambda x, y: jnp.sin(np.pi * y) * jnp.cos(np.pi * x)))
        Fx = F.diffx()
        Fy = F.diffy()
        ident = (Fx.components[0] * Fy.components[1]
                 - Fy.components[0] * Fx.components[1])
        assert _norm_inf(F.jacobian() - ident) < 1e5 * TOL
        true = _dsk(lambda x, y:
                    -np.pi ** 2 * y * jnp.sin(np.pi * x * y)
                    * jnp.cos(np.pi * y) * jnp.cos(np.pi * x)
                    - np.pi ** 2 * x * jnp.sin(np.pi * x * y)
                    * jnp.sin(np.pi * y) * jnp.sin(np.pi * x))
        assert _norm_inf(F.jacobian() - true) < 4e6 * TOL
