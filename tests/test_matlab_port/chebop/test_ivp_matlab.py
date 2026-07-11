"""Port of MATLAB Chebfun tests/chebop/test_ivp.m (Fable 5).

u' - u = 1 - x, u(-1) = exp(-1) - 1  ->  u = exp(x) + x.

Provenance
----------
MATLAB source : tests/chebop/test_ivp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

TOL = 1e-8


class TestChebopIvp:
    def test_linear_ivp_exact_solution(self):
        A = Chebop(lambda x, u: u.diff() - u)
        A.lbc = float(np.exp(-1) - 1)
        rhs = cj.chebfun(lambda x: 1 - x)
        u = A.solve(rhs)
        xs = jnp.asarray(np.linspace(-0.95, 0.95, 40))
        exact = jnp.exp(xs) + xs
        err = jnp.abs(u(xs) - exact)
        assert float(jnp.max(err)) < TOL
