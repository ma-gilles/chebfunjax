"""Port of MATLAB Chebfun tests/chebop/test_ellipjODE.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_ellipjODE.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from scipy.special import ellipk

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

TOL = 1e-10


def _ninf(f, d):
    xs = jnp.linspace(d[0], d[1], 401)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


class TestChebopEllipjODE:
    def test_all_matlab_assertions(self):
        m = 0.1
        K = float(ellipk(m))
        d = (0.0, K)
        x = cj.chebfun(lambda t: t, domain=d)
        sn, cn, dn = x.ellipj(m)

        # err(1): sn solves u'' + (1+m) u - 2 m u^3 = 0, u(0)=0, u(K)=1
        N = Chebop(lambda x, u: u.diff(2) + (1 + m) * u - 2 * m * u ** 3,
                   domain=d)
        N.lbc = lambda u: u
        N.rbc = lambda u: u - 1.0
        u = N.solve(0.0)
        assert _ninf(u - sn, d) < TOL

        # err(2): cn solves u'' + (1-2m) u + 2 m u^3 = 0, u(0)=1, u(K)=0
        N = Chebop(lambda x, u: u.diff(2) + (1 - 2 * m) * u + 2 * m * u ** 3,
                   domain=d)
        N.lbc = lambda u: u - 1.0
        N.rbc = lambda u: u
        u = N.solve(0.0)
        assert _ninf(u - cn, d) < TOL
