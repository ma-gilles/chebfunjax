"""Port of MATLAB Chebfun tests/chebop/test_firstOrderIntegralEqn.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_firstOrderIntegralEqn.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)


def _n(f, d, n=33):
    xs = jnp.linspace(d[0] + 1e-9, d[1] - 1e-9, n)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


class TestChebopFirstOrderIntegralEqn:
    def test_all_matlab_assertions(self):
        d = (0.0, 5.0)
        L = Chebop(lambda u: u.diff() + 2.0 * u + 5.0 * u.cumsum(),
                   domain=d)
        L.lbc = 0.0
        u1 = L.solve(1.0)
        L2 = Chebop(lambda u: u.diff() + 2.0 * u + 5.0 * u.cumsum(),
                    domain=d)
        L2.bc = lambda x, u: u(0.0)
        u2 = L2.solve(1.0)
        assert _n(u1 - u2, d) < 1e-10                      # pass(1)

        # pass(2): u' + 2*sum((5-x) sin(x) u) = sin(pi x), u(5) = 1
        L = Chebop(
            lambda x, u: u.diff()
            + 2.0 * ((5.0 - x) * x.sin() * u).sum(), domain=d)
        L.rbc = 1.0
        x = cj.chebfun(lambda t: t, domain=d)
        rhs = (jnp.pi * x).sin()
        u = L.solve(rhs)
        assert _n(L * u - rhs, d) < 1e-10
        assert abs(float(u(jnp.asarray(5.0))) - 1.0) < 1e-10
