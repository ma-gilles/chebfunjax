"""Literal complete ten-clause port of MATLAB tests/chebop/test_bc.m.

Function residuals use scalar Chebfun L2 norms; two-component boundary
residuals use Euclidean vector norms. Every source bound is 1e-10.

Provenance
----------
MATLAB source : tests/chebop/test_bc.m, @chebop/solvebvp.m,
    @chebop/feval.m, @chebop/subsref.m, @chebfun/norm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

TOL = 1e-10


class TestChebopBc:
    def test_all_matlab_assertions(self, record_property):
        errors = []

        def record(value):
            value = float(value)
            errors.append(value)
            record_property(f"source_bc_err_{len(errors)}", value)

        # Source1–3: numeric left BC and literal string right Neumann BC.
        d = (-3.0, 4.0)
        A = Chebop(lambda x, u: u.diff(2) + 4*u.diff() + u, domain=d)
        A.lbc = -1
        A.rbc = "neumann"
        f = cj.chebfun(lambda x: jnp.exp(jnp.sin(x)), domain=d)
        # Python's public solvebvp always returns (u, info).
        u, _ = A.solvebvp(f)
        record((u.diff(2) + 4*u.diff() + u - f).norm())
        record(abs(u(d[0]) + 1))
        record(abs(u.diff()(d[1])))

        # Source4–6: Robin and Neumann function-handle boundary conditions.
        d = (-1.0, 0.0)
        A = Chebop(lambda x, u: u.diff(2) + 4*u.diff() + 200*u, domain=d)
        A.lbc = lambda u: [u.diff() + 2*u - 1]
        A.rbc = lambda u: u.diff()
        f = cj.chebfun(lambda x: x*jnp.sin(3*x)**2, domain=d)
        u, _ = A.solvebvp(f)
        du = u.diff()
        record((u.diff(2) + 4*u.diff() + 200*u - f).norm())
        record(abs(du(d[0]) + 2*u(d[0]) - 1))
        record(abs(u.diff()(d[1])))

        # Source7–8: interior derivative evaluation plus global integral.
        dom = (-2.0, 1.0)
        N = Chebop(lambda x, u: u.diff(2) + u, domain=dom)
        N.bc = lambda x, u: [u.diff()(0.0), u.sum()]
        x = cj.chebfun(lambda x: x, domain=dom)
        rhs = x.sin()
        u, _ = N.solvebvp(rhs)
        record((N(x, u) - rhs).norm())
        record(jnp.linalg.norm(jnp.asarray(N.bc(x, u)), ord=2))

        # Source9–10: same exotic constraints for a nonlinear operator.
        dom = (-1.0, 1.0)
        N = Chebop(lambda x, u: u.diff(2) + u.sin(), domain=dom)
        N.bc = lambda x, u: [u.diff()(0.0), u.sum()]
        x = cj.chebfun(lambda x: x, domain=dom)
        rhs = x.sin()
        u, _ = N.solvebvp(rhs)
        record((N(x, u) - rhs).norm())
        record(jnp.linalg.norm(jnp.asarray(N.bc(x, u)), ord=2))

        assert len(errors) == 10
        assert all(error < TOL for error in errors), {
            f"source_bc_err_{case}": error
            for case, error in enumerate(errors, 1)
        }
