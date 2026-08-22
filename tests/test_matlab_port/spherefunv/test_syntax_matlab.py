"""Port of MATLAB Chebfun tests/spherefunv/test_syntax.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_syntax.m
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




class TestSpherefunvSyntax:
    def test_all_matlab_assertions(self):
        for jj in (1, 2):
            f = lambda x, y, z: jj * jnp.sin(x * y * z)
            g_sph = lambda lam, th: jnp.exp(
                (jnp.cos(lam) * jnp.sin(th)) ** jj)
            h = lambda x, y, z: jnp.cos(jj * x) + f(x, y, z)

            fs = _sph(f)
            gs = Spherefun.from_function(g_sph)
            hs = _sph(h)

            F1 = Spherefunv.from_functions(
                lambda lam, th: fs(lam, th),
                lambda lam, th: gs(lam, th),
                lambda lam, th: hs(lam, th))
            F3 = Spherefunv(fs, gs, hs)
            for a, b in zip(F1.components, F3.components):
                assert _norm_inf(a - b) < 10 * TOL
