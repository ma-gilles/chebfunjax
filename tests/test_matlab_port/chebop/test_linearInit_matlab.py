"""Port of MATLAB Chebfun tests/chebop/test_linearInit.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_linearInit.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

TOL = 1e-8


def _n(f, d):
    xs = jnp.linspace(d[0] + 1e-9, d[1] - 1e-9, 33)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


class TestChebopLinearInit:
    def test_all_matlab_assertions(self):
        import math
        d = (0.0, math.pi)
        N = Chebop(lambda x, u: u.diff(2) + x * u, domain=d)
        N.lbc = 2.0
        N.rbc = 3.0
        x = cj.chebfun(lambda t: t, domain=d)
        rhs = x.sin()
        # A wild init must not derail the LINEAR solve.
        N.init = (20.0 * x).sin()
        u1 = N.solve(rhs)
        assert _n(N(u1) - rhs, d) < TOL
        assert abs(float(u1(jnp.asarray(0.0))) - 2.0) < TOL
        assert abs(float(u1(jnp.asarray(math.pi))) - 3.0) < TOL
