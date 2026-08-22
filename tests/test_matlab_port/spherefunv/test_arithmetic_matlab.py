"""Port of MATLAB Chebfun tests/spherefunv/test_arithmetic.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_arithmetic.m
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




class TestSpherefunvArithmetic:
    def test_all_matlab_assertions(self):
        f = lambda x, y, z: jnp.cos(2 * np.pi * x * y * z)
        g = lambda x, y, z: jnp.sin(2 * np.pi * x * y * z)
        h = lambda x, y, z: 1 + x * y * z
        u = Spherefunv(_sph(f), _sph(g), _sph(h))
        v = Spherefunv(_sph(h), _sph(f), _sph(g))
        plus = Spherefunv(
            _sph(lambda x, y, z: f(x, y, z) + h(x, y, z)),
            _sph(lambda x, y, z: g(x, y, z) + f(x, y, z)),
            _sph(lambda x, y, z: h(x, y, z) + g(x, y, z)))
        minus = Spherefunv(
            _sph(lambda x, y, z: f(x, y, z) - h(x, y, z)),
            _sph(lambda x, y, z: g(x, y, z) - f(x, y, z)),
            _sph(lambda x, y, z: h(x, y, z) - g(x, y, z)))
        mult = Spherefunv(
            _sph(lambda x, y, z: f(x, y, z) * h(x, y, z)),
            _sph(lambda x, y, z: g(x, y, z) * f(x, y, z)),
            _sph(lambda x, y, z: h(x, y, z) * g(x, y, z)))
        assert _vnorm(u + v - plus) < 10 * TOL  # pass(1)
        assert _vnorm(u - v - minus) < 10 * TOL  # pass(2)
        assert _vnorm(u.times(v) - mult) < 100 * TOL  # pass(3)
