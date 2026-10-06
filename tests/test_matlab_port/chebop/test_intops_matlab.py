"""Literal ten-clause port of MATLAB tests/chebop/test_intops.m.

Default scalar norms are L2; explicit infinity norms use continuous extrema.
Legacy sampled maxima are diagnostic properties, not source predicates.

Provenance
----------
MATLAB source : tests/chebop/test_intops.m, @chebop/subsref.m,
    @chebfun/norm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop
from chebfunjax.operators.integral import fred, volt

jax.config.update("jax_enable_x64", True)

TOL = 1e-10


def _sampled_max_diagnostic(f, d, n=33):
    """Preserve the old port's metric solely as an observed diagnostic."""
    xs = jnp.linspace(d[0] + 1e-9, d[1] - 1e-9, n)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


class TestChebopIntops:
    def test_all_matlab_assertions(self, record_property):
        # Source initializes pass from err(1:4) < tol and later assigns
        # pass(5:10). Evaluate every clause before checking the vector.
        passes = []

        def check(case, error, domain, *, p=2):
            value = float(error.norm(p))
            passes.append(value < TOL)
            record_property(f"source_case_{case}_norm", value)
            record_property(f"source_case_{case}_norm_order", p)
            record_property(
                f"diagnostic_case_{case}_legacy_sampled_max",
                _sampled_max_diagnostic(error, domain),
            )

        # Source1: general BC u(0), not the old scalar-lbc adapter.
        d = (0.0, 5.0)
        x = cj.chebfun(lambda t: t, domain=d)
        N = Chebop(lambda u: u.diff() + 2*u + 5*u.cumsum(), domain=d)
        N.bc = lambda x, u: u(0.0)
        u = N.solve(1.0)
        u_exact = 0.5*(-x).exp()*(2*x).sin()
        check(1, u-u_exact, d)

        # Source2: Fredholm.
        d = (0.0, 1.0)
        x = cj.chebfun(lambda t: t, domain=d)
        K = lambda x, y: jnp.sin(2*jnp.pi*(x-y))
        A = Chebop(lambda x, u: u + fred(K, u), domain=d)
        u = x*x.exp()
        f = A*u
        check(2, u-A.solve(f), d)

        # Source3/4: Volterra solution and operator residual, both L2.
        d = (0.0, float(jnp.pi))
        x = cj.chebfun(lambda t: t, domain=d)
        K = lambda x, y: x*y
        A = Chebop(lambda x, u: u-volt(K, u), domain=d)
        f = x**2*x.cos() + (1-x)*x.sin()
        u = A.solve(f)
        check(3, u-x.sin(), d)
        check(4, A*u-f, d)

        # Source5 assigns a NUMBER into the already logical pass vector.
        # It contains no <tol. bool(value) preserves nonzero-to-true casting;
        # the recorded value must not be described as a tolerance pass.
        K = lambda x, t: jnp.exp(t*(x-t))
        N = Chebop(lambda x, y: y.diff() + y - x*(1+2*x)*volt(K, y) - 1 - 2*x)
        N.lbc = 1
        y = N.solve(0.0)
        # MATLAB N.op(y) is dispatched through @chebop/subsref.m:28–30
        # to feval(N,y). Python N(y) is the corresponding public adapter.
        residual_norm = N(y).norm()
        interior_difference = abs(y(0.0)-1)
        value = float(residual_norm + interior_difference)
        record_property("source_case_5_equation_residual_l2", float(residual_norm))
        record_property("source_case_5_interior_abs_y0_minus1", float(interior_difference))
        record_property("source_case_5_domain", tuple(N.domain))
        record_property("source_case_5_left_bc_residual", float(y(-1.0)-1))
        passes.append(bool(value))
        record_property("source_case_5_numeric_logical_assignment", value)

        # Source6: default [-1,1] domain, explicit infinity norm.
        K = lambda x, y: jnp.exp(-(x-y))
        A = Chebop(lambda x, u: u.diff() + fred(K, u))
        A.lbc = 0
        u = A.solve(1.0)
        check(6, A*u-1, A.domain, p=jnp.inf)

        # Source7: nonlocal sum functional, explicit infinity norm.
        d = (0.0, 1.0)
        L = Chebop(lambda u: u.diff() + u.sum(), domain=d)
        L.lbc = 1
        u = L.solve(0.0)
        x = cj.chebfun(lambda t: t, domain=d)
        check(7, u-(1-2*x/3), d, p=jnp.inf)

        # Source8: nonlinear periodic operator, default domain and L2.
        A = Chebop(lambda x, u: u.diff(2) + (2*jnp.pi*x).sin()*u + u.sum()*u)
        A.bc = "periodic"
        u = A.solve(1.0)
        check(8, A*u-1, A.domain)

        # Source9: periodic integro-differential operator, default L2.
        f = cj.chebfun(lambda x: jnp.sin(5*jnp.pi*x))
        K = lambda x, y: jnp.cos(-jnp.pi*(x-y))
        A = Chebop(lambda x, u: (1/25)*u.diff(2) + fred(K, u))
        A.bc = "periodic"
        u = A.solve(f)
        check(9, A*u-f, A.domain)

        # Source10: nonlinear Volterra equation, default L2.
        K = lambda x, y: 1+0*x
        N = Chebop(
            lambda x, u: -u + x.exp() + (1/3)*x*(1-(3*x).exp())
            + x*volt(K, u**3),
            domain=(0.0, 1.0),
        )
        u = N.solve(0.0)
        check(10, u-cj.chebfun(jnp.exp, domain=(0.0, 1.0)), N.domain)

        assert len(passes) == 10
        assert all(passes), {
            f"source_case_{case}": passed
            for case, passed in enumerate(passes, 1)
        }
