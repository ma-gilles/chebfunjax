"""Port of MATLAB Chebfun tests/chebop/test_bcVectorInput.m (Fable 5).

Vector-valued BC specs: for a scalar problem ``N.lbc = [v0; v1; ...]``
imposes the derivative ladder u(a)=v0, u'(a)=v1, ...; for a first-order
system the entries are per-component values.

Provenance
----------
MATLAB source : tests/chebop/test_bcVectorInput.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)


def _d(u, k, x):
    return float(u.diff(k)(jnp.asarray(x))) if k else \
        float(u(jnp.asarray(x)))


class TestChebopBcvectorinput:
    def test_second_order_ivp(self):
        N = Chebop(lambda x, u: u.diff(2) + u ** 2, (0.5, 2.0))
        N.lbc = [0.1, -1.0]
        u = N.solve(0.0)
        assert (abs(_d(u, 0, 0.5) - 0.1)
                + abs(_d(u, 1, 0.5) + 1)) < 5e-10  # pass(1)

    def test_both_lbc_rbc_scalar(self):
        N = Chebop(lambda x, u: u.diff(2) + u ** 2, (0.5, 2.0))
        N.lbc = 0.2
        N.rbc = -0.2
        u = N.solve(0.0)
        assert (abs(_d(u, 0, 0.5) - 0.2)
                + abs(_d(u, 0, 2.0) + 0.2)) < 1e-10  # pass(2)

    def test_fourth_order_ivp(self):
        N = Chebop(lambda x, u: u.diff(4) + u.sin(), (0.6, 3.0))
        N.lbc = [0.1, 1.0, 3.2, -2.0]
        u = N.solve(0.0)
        err = (abs(_d(u, 0, 0.6) - 0.1) + abs(_d(u, 1, 0.6) - 1)
               + abs(_d(u, 2, 0.6) - 3.2) + abs(_d(u, 3, 0.6) + 2))
        assert err < 2e-7  # pass(3)

    def test_fourth_order_bvp(self):
        N = Chebop(lambda x, u: u.diff(4) + u.sin(), (0.6, 3.0))
        N.lbc = [0.1, 1.0]
        N.rbc = [-0.1, 1.0]
        u = N.solve(0.0)
        err = (abs(_d(u, 0, 0.6) - 0.1) + abs(_d(u, 1, 0.6) - 1)
               + abs(_d(u, 0, 3.0) + 0.1) + abs(_d(u, 1, 3.0) - 1))
        assert err < 1e-11  # pass(4)

    def test_bc_vector_both_ends(self):
        N = Chebop(lambda x, u: u.diff(4) + u.sin(), (0.6, 3.0))
        N.bc = [0.2, 0.8]
        u = N.solve(0.0)
        err = (abs(_d(u, 0, 0.6) - 0.2) + abs(_d(u, 1, 0.6) - 0.8)
               + abs(_d(u, 0, 3.0) - 0.2) + abs(_d(u, 1, 3.0) - 0.8))
        assert err < 1e-11  # pass(5)

    def test_first_order_system(self):
        N = Chebop(lambda x, u, v: [u.diff() + v, v.diff() - u],
                   (-1.0, 1.0))
        N.lbc = [1.0, 2.0]
        sol = N.solve([0, 0])
        u, v = sol[0], sol[1]
        assert (abs(_d(u, 0, -1.0) - 1)
                + abs(_d(v, 0, -1.0) - 2)) < 1e-10  # pass(6)
