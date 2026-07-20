"""Port of MATLAB Chebfun tests/ballfun/test_sum.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_sum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun

_EPS = float(np.finfo(np.float64).eps)
_TOL = 1e4 * _EPS


def _bf(fn):
    return Ballfun.from_function(fn, spherical=True)


class TestBallfunSum:
    def test_integrate_over_r_example1(self):
        # f = r cos(lam) sin(th); sum(f, 1) = spherefun(cos(lam) sin(th) / 4).
        f = _bf(lambda r, lam, th: r * jnp.cos(lam) * jnp.sin(th))
        g = f.sum(1)
        lam = np.linspace(-np.pi, np.pi, 24)
        th = np.linspace(0.02, np.pi - 0.02, 19)
        LL, TT = np.meshgrid(lam, th)
        got = np.asarray(g(jnp.asarray(LL), jnp.asarray(TT)))
        exact = np.cos(LL) * np.sin(TT) / 4.0
        assert np.max(np.abs(got - exact)) < _TOL

    def test_integrate_over_r_example2(self):
        # f = 1; sum(f, 1) = spherefun(1/3).
        f = _bf(lambda r, lam, th: jnp.ones_like(r))
        g = f.sum(1)
        lam = np.linspace(-np.pi, np.pi, 12)
        th = np.linspace(0.1, np.pi - 0.1, 9)
        LL, TT = np.meshgrid(lam, th)
        got = np.asarray(g(jnp.asarray(LL), jnp.asarray(TT)))
        assert np.max(np.abs(got - 1.0 / 3.0)) < _TOL

    def test_integrate_over_lambda_example3(self):
        # f = (r sin(lam) sin(th))^2; sum(f, 2) = diskfun(pi r^2 sin(th)^2).
        f = _bf(lambda r, lam, th: (r * jnp.sin(lam) * jnp.sin(th)) ** 2)
        g = f.sum(2)
        th = np.linspace(-np.pi, np.pi, 24)
        r = np.linspace(0.05, 0.95, 17)
        TH, R = np.meshgrid(th, r)
        got = np.asarray(g(jnp.asarray(TH), jnp.asarray(R)))
        exact = np.pi * R**2 * np.sin(TH) ** 2
        assert np.max(np.abs(got - exact)) < _TOL

    def test_integrate_over_theta_example4(self):
        # f = r cos(lam) sin(th); sum(f, 3) = diskfun(r cos(lam) pi/2).
        f = _bf(lambda r, lam, th: r * jnp.cos(lam) * jnp.sin(th))
        g = f.sum(3)
        lam = np.linspace(-np.pi, np.pi, 24)
        r = np.linspace(0.05, 0.95, 17)
        LAM, R = np.meshgrid(lam, r)
        got = np.asarray(g(jnp.asarray(LAM), jnp.asarray(R)))
        exact = R * np.cos(LAM) * np.pi / 2.0
        assert np.max(np.abs(got - exact)) < _TOL
