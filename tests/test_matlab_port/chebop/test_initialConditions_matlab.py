"""Port of MATLAB Chebfun tests/chebop/test_initialConditions.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_initialConditions.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

TOL = 5e-9
D = (0.0, 1.0)


def _v(f, x):
    return float(f(jnp.asarray(x)))


class TestChebopInitialConditions:
    def test_scalar_conditions(self):
        for op in (lambda x, u: u.diff() + u.sin(),
                   lambda u: u.diff() + u.sin()):
            N = Chebop(op, domain=D)
            N.lbc = lambda u: u - 0.5
            u = N.solve(0.0)
            assert abs(_v(u, 0.0) - 0.5) < TOL       # pass(1)/(5)
            N = Chebop(op, domain=D)
            N.rbc = lambda u: u - 0.5
            u = N.solve(0.0)
            assert abs(_v(u, 1.0) - 0.5) < TOL       # pass(2)/(6)

        for op in (lambda x, u: u.diff(2) + u.sin(),
                   lambda u: u.diff(2) + u.sin()):
            N = Chebop(op, domain=D)
            N.lbc = lambda u: [u - 0.5, u.diff() - 1.3]
            u = N.solve(0.0)
            assert (abs(_v(u, 0.0) - 0.5)
                    + abs(_v(u.diff(), 0.0) - 1.3)) < TOL  # pass(3)/(7)
            N = Chebop(op, domain=D)
            N.rbc = lambda u: [u - 0.5, u.diff() - 1.3]
            u = N.solve(0.0)
            assert (abs(_v(u, 1.0) - 0.5)
                    + abs(_v(u.diff(), 1.0) - 1.3)) < TOL  # pass(4)/(8)

    def test_system_conditions(self):
        op = lambda x, u, v, w: [u.diff() + v, v.diff() + w, w.diff() + u]
        N = Chebop(op, domain=D)
        N.lbc = lambda u, v, w: [u - 0.5, v - 1.5, w + 2.5]
        U = N.solve([0.0, 1.0, 2.0])
        assert (abs(_v(U[0], 0.0) - 0.5) + abs(_v(U[1], 0.0) - 1.5)
                + abs(_v(U[2], 0.0) + 2.5)) < TOL     # pass(9)
        N = Chebop(op, domain=D)
        N.rbc = lambda u, v, w: [u - 0.5, v - 1.5, w + 2.5]
        U = N.solve([0.0, 1.0, 2.0])
        assert (abs(_v(U[0], 1.0) - 0.5) + abs(_v(U[1], 1.0) - 1.5)
                + abs(_v(U[2], 1.0) + 2.5)) < TOL     # pass(10)

    def test_third_order_system(self):
        op = lambda x, u, v, w: [u.diff(3) + v, v.diff(3) + w,
                                 w.diff(3) + u]
        N = Chebop(op, domain=D)
        N.lbc = lambda u, v, w: [
            u - 0.5, v - 1.5, w + 2.5,
            u.diff() - 1.0, v.diff() - 2.0, w.diff() + 1.0,
            u.diff(2), v.diff(2) - 0.5, w.diff(2) + 0.5]
        U = N.solve([0.0, 1.0, 2.0])
        err = (abs(_v(U[0], 0.0) - 0.5) + abs(_v(U[1], 0.0) - 1.5)
               + abs(_v(U[2], 0.0) + 2.5)
               + abs(_v(U[0].diff(), 0.0) - 1.0)
               + abs(_v(U[1].diff(), 0.0) - 2.0)
               + abs(_v(U[2].diff(), 0.0) + 1.0)
               + abs(_v(U[0].diff(2), 0.0))
               + abs(_v(U[1].diff(2), 0.0) - 0.5)
               + abs(_v(U[2].diff(2), 0.0) + 0.5))
        assert err < 100 * TOL                        # pass(11)
