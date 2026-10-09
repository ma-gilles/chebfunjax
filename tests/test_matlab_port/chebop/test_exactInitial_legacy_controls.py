"""Port of MATLAB Chebfun tests/chebop/test_exactInitial.m (Fable 5).

The chebcolloc1/ultraS discretization variants run through
solve_bvp_altdisc.

Provenance
----------
MATLAB source : tests/chebop/test_exactInitial.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop
from chebfunjax.operators.chebop_altdisc import solve_bvp_altdisc

jax.config.update("jax_enable_x64", True)

TOL = 1e-7


def _n(f, d):
    xs = jnp.linspace(d[0] + 1e-9, d[1] - 1e-9, 33)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


class TestChebopExactInitial:
    def test_all_matlab_assertions(self):
        d = (0.0, 10.0)
        x = cj.chebfun(lambda t: t, domain=d)

        def mk():
            N = Chebop(lambda x, u: u.diff(2) + u.sin(), domain=d)
            N.lbc = 2.0
            N.rbc = 2.0
            return N

        N = mk()
        N.init = 2.0 * (2.0 * jnp.pi * x / 10.0).cos()
        u = N.solve(0.0)
        assert _n(N(u), d) < TOL           # err(1)

        # Restart from the exact solution.
        N = mk()
        N.init = u
        u = N.solve(0.0)
        assert _n(N(u), d) < TOL           # err(2)

        # chebcolloc1 / ultraS variants.
        for disc in ("chebcolloc1", "ultraS"):
            N = mk()
            N.init = u
            v = solve_bvp_altdisc(N, 0.0, disc, n=64)[0]
            assert _n(N(v), d) < TOL       # err(3)/(4)
