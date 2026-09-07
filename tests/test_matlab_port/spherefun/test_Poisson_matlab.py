"""Port of MATLAB Chebfun tests/spherefun/test_Poisson.m (Fable 5).

``spherefun.poisson(f, const, m, n)`` is ``Spherefun.poisson(f, const,
m, n)``; ``norm(u - exact, inf)`` is ``(u - exact).norm('inf')``.

Provenance
----------
MATLAB source : tests/spherefun/test_Poisson.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun

jax.config.update("jax_enable_x64", True)


def _sph(fn):
    return Spherefun.from_function(fn)


def _xyz(fn):
    return Spherefun.from_function(
        lambda lam, th: fn(jnp.cos(lam) * jnp.sin(th), jnp.sin(lam) * jnp.sin(th),
                           jnp.cos(th)))


class TestSpherefunPoisson:
    def test_all_matlab_assertions(self):
        m = 40
        n = 40
        tol = 1e3 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = _sph(lambda lam, th: -6 * jnp.cos(lam) * jnp.cos(th) * jnp.sin(th))
        exact = _sph(lambda lam, th: jnp.sin(th) * jnp.cos(th) * jnp.cos(lam))
        u = Spherefun.poisson(f, 0, m, n)
        assert float((u - exact).norm("inf")) < tol                     # pass(1)

        f = _sph(lambda lam, th: -4 * (3 * jnp.cos(th) + 5 * jnp.cos(3 * th))
                 * jnp.sin(lam) * jnp.sin(th))
        exact = _sph(lambda lam, th: -2 * jnp.sin(lam) * jnp.sin(2 * th)
                     * jnp.sin(th) ** 2)
        u = Spherefun.poisson(f, 0, m, n)
        assert float((u - exact).norm("inf")) < tol                     # pass(2)

        f = _sph(lambda lam, th: -6 * (-1 + 5 * jnp.cos(2 * th)) * jnp.sin(lam)
                 * jnp.sin(2 * th))
        exact = _sph(lambda lam, th: -2 * jnp.sin(lam) * jnp.sin(2 * th)
                     * jnp.sin(th) ** 2 - jnp.sin(lam) * jnp.sin(th) * jnp.cos(th)
                     + .5 * jnp.sin(lam) * jnp.sin(2 * th) * jnp.cos(2 * th))
        u = Spherefun.poisson(f, 0, m, n)
        assert float((u - exact).norm("inf")) < tol                     # pass(3)
        assert abs(float(u.mean2())) < tol                              # pass(4)

        f = _xyz(lambda x, y, z: 1 + x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = Spherefun.poisson(f, 0, 10)
        g = _xyz(lambda x, y, z: x)
        v = Spherefun.poisson(g, 0, 10)
        assert float((u - v).norm()) < tol                              # pass(5)

        f = _xyz(lambda x, y, z: x * y * z)
        u = Spherefun.poisson(f, 1, 10)
        assert abs(float(u.mean2()) - 1) < tol                          # pass(6)
