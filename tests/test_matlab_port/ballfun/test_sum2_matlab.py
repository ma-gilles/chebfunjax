"""Port of MATLAB Chebfun tests/ballfun/test_sum2.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_sum2.m
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


class TestBallfunSum2:
    def test_lambda_theta_constant(self):
        # f = 1; sum2(f, [2, 3]) = chebfun(4 pi) in r.
        f = _bf(lambda r, lam, th: jnp.ones_like(r))
        g = f.sum2((2, 3))
        r = np.linspace(0.0, 1.0, 15)
        assert np.max(np.abs(np.asarray(g(jnp.asarray(r))) - 4.0 * np.pi)) < _TOL

    def test_theta_lambda_zero(self):
        # f = r cos(lam) sin(th); sum2(f, [3, 2]) = chebfun(0).
        f = _bf(lambda r, lam, th: r * jnp.cos(lam) * jnp.sin(th))
        g = f.sum2((3, 2))
        r = np.linspace(0.0, 1.0, 15)
        assert np.max(np.abs(np.asarray(g(jnp.asarray(r))))) < _TOL

    def test_r_theta_trig_in_lambda(self):
        # f = r sin(th) cos(lam); sum2(f, [1, 3]) = cos(lam) pi/8, trig on [-pi,pi].
        f = _bf(lambda r, lam, th: r * jnp.sin(th) * jnp.cos(lam))
        g = f.sum2((1, 3))
        lam = np.linspace(-np.pi, np.pi, 23)
        got = np.asarray(g(jnp.asarray(lam)))
        assert np.max(np.abs(got - np.cos(lam) * np.pi / 8.0)) < _TOL
