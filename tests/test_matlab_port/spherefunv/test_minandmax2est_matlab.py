"""Port of MATLAB Chebfun tests/spherefunv/test_minandmax2est.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_minandmax2est.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

jax.config.update("jax_enable_x64", True)

TOL = 1e3 * 2.220446049250313e-16


def _sph(fc):
    def f(lam, th):
        x = jnp.cos(lam) * jnp.sin(th)
        y = jnp.sin(lam) * jnp.sin(th)
        z = jnp.cos(th)
        return fc(x, y, z)
    return Spherefun.from_function(f)


def _norm_inf(g, n=25):
    lam = jnp.linspace(-np.pi + 1e-6, np.pi - 1e-6, n)
    th = jnp.linspace(1e-3, np.pi - 1e-3, n)
    L, T = jnp.meshgrid(lam, th)
    return float(jnp.max(jnp.abs(jnp.asarray(g(L, T)))))


def _vnorm(F, n=25):
    return max(_norm_inf(c, n) for c in F.components)


class TestSpherefunvMinandmax2est:
    def test_all_matlab_assertions(self):
        F = Spherefunv(_sph(lambda x, y, z: x),
                       _sph(lambda x, y, z: y),
                       _sph(lambda x, y, z: z))
        r = F.minandmax2est()
        box = [-1, 1, -1, 1, -1, 1]
        # isSubset: estimated range within the true box (tolerance).
        for k in range(3):
            assert r[2 * k] >= box[2 * k] - TOL
            assert r[2 * k + 1] <= box[2 * k + 1] + TOL
