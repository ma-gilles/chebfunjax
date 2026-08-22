"""Port of MATLAB Chebfun tests/spherefunv/test_times_divide.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_times_divide.m
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


class TestSpherefunvTimesDivide:
    def test_all_matlab_assertions(self):
        f = _sph(lambda x, y, z: jnp.sin(y * z) + jnp.cos(2 * x * y))
        F = Spherefunv(f, f, f)
        G = Spherefunv(2 * f, 2 * f, 2 * f)
        H = Spherefunv(f * 0.5, f * 0.5, f * 0.5)
        assert _vnorm(2 * F - G) < TOL  # pass(1)/(2)
        assert _vnorm(F * 2 - G) < TOL  # pass(3)/(4)
        assert _vnorm(F * 0.5 - H) < TOL  # pass(5)/(6): F/2
