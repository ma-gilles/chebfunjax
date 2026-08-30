"""Port of MATLAB Chebfun tests/chebop/test_pcg.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_pcg.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

TOL = 1e2 * 1e-8   # 1e2 * cheboppref bvpTol


def _n(f, d=(-1.0, 1.0)):
    xs = jnp.linspace(d[0] + 1e-9, d[1] - 1e-9, 33)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


def _mkop(a, c, dom=(-1.0, 1.0), b=None):
    if b is None:
        op = lambda x, u: (-1.0) * (a(x) * u.diff()).diff() + c(x) * u
    else:
        op = lambda x, u: ((-1.0) * (a(x) * u.diff()).diff()
                           + b(x) * u.diff() + c(x) * u)
    return Chebop(op, domain=dom)


class TestChebopPcg:
    def test_all_matlab_assertions(self):
        cases = [
            (lambda x: 1.0 + 0 * x, lambda x: 1.0 + 0 * x),
            (lambda x: 2 + (jnp.pi * x).cos(), lambda x: 0 * x),
            (lambda x: 2 + (jnp.pi * x).cos(), lambda x: 1 + 10 * x ** 2),
        ]
        f = cj.chebfun(lambda x: 1 - 3 * x ** 2)
        for a, c in cases:
            N = _mkop(a, c)
            N.bc = 0.0
            u = N.solve(f)
            v = N.pcg(f)
            assert _n(u - v) < TOL

        # pass(5): inhomogeneous Dirichlet values.
        N = _mkop(lambda x: 2 + (jnp.pi * x).cos(),
                  lambda x: 1 + 10 * x ** 2)
        N.lbc = 1.0
        N.rbc = -1.0
        f = cj.chebfun(lambda x: 1 - 2 * x ** 2)
        u = N.solve(f)
        v = N.pcg(f)
        assert _n(u - v) < TOL
