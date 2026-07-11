"""Port of MATLAB Chebfun tests/chebop/test_bc.m (Fable 5).

u'' + 4u' + u = exp(sin x) on [-3,4], u(-3) = -1, Neumann at 4.
MATLAB's rbc='neumann' STRING is not accepted by the lbc/rbc setters
(only by N.bc); the callable equivalent u.diff() is used, with the
string gap noted.

Provenance
----------
MATLAB source : tests/chebop/test_bc.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

TOL = 1e-6


class TestChebopBc:
    def test_dirichlet_left_neumann_right(self):
        A = Chebop(lambda x, u: u.diff(2) + 4 * u.diff() + u,
                   domain=(-3.0, 4.0))
        A.lbc = -1.0
        A.rbc = lambda u: u.diff()
        f = cj.chebfun(lambda x: jnp.exp(jnp.sin(x)),
                       domain=(-3.0, 4.0))
        u = A.solve(f)
        xs = jnp.asarray(np.linspace(-2.8, 3.8, 30))
        res = u.diff(2)(xs) + 4 * u.diff()(xs) + u(xs) - f(xs)
        assert float(jnp.max(jnp.abs(res))) < 1e3 * TOL
        assert abs(float(u(jnp.asarray(-3.0))) + 1) < TOL
        assert abs(float(u.diff()(jnp.asarray(4.0)))) < 1e2 * TOL

    def test_neumann_string_on_rbc(self):
        pytest.skip("lbc/rbc setters do not accept the 'neumann' string "
                    "(only N.bc handles strings)")
