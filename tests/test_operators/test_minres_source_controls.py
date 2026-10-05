"""Independent analytic controls for the source MINRES function-space port.

Provenance
----------
MATLAB source : @chebop/minres.m, tests/chebop/test_minres.m
Chebfun commit: 7574c77
The indefinite operator also occurs in examples/ode-linear/Krylov.m.
"""

import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.operators import krylov
from chebfunjax.operators.chebop import Chebop

# Same solution-error bound as the pinned five MATLAB tests:100*bvpTol.
SOURCE_TOL = 5e-11
# Independent diagnostic bound for twice-differentiated analytic controls.
# This is not a replacement or widening of any original MATLAB assertion.
RESIDUAL_TOL = 1e-9


def test_minres_uses_lanczos_not_fixed_grid_arnoldi(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("source MINRES must not call fixed-grid Arnoldi")

    monkeypatch.setattr(krylov, "_arnoldi_solve", forbidden)
    op = Chebop(lambda u: -u.diff(2))
    op.bc = 0
    f = cj.chebfun(lambda x: 1 - 3 * x**2)
    exact = cj.chebfun(lambda x: (1 - x**2)**2 / 4)
    actual = krylov.minres(op, f)
    assert float((actual - exact).norm()) < SOURCE_TOL


@pytest.mark.parametrize("domain", [(-1., 1.), (-1., 0., 1.)])
def test_minres_indefinite_constant_rhs(domain):
    op = Chebop(lambda u: -u.diff(2) - 100 * u, domain=domain)
    op.bc = 0
    f = cj.chebfun(lambda x: 1 + 0*x, domain=domain)
    exact = cj.chebfun(lambda x: (jnp.cos(10*x)/jnp.cos(10.) - 1)/100,
                       domain=domain)
    actual = krylov.minres(op, f, tol=5e-13, maxit=40)
    assert float((actual - exact).norm()) < SOURCE_TOL
    assert float((op(actual) - f).norm()) / float(f.norm()) < RESIDUAL_TOL
    assert float(jnp.max(jnp.abs(actual(jnp.array([-1., 1.]))))) < SOURCE_TOL


def test_minres_indefinite_piecewise_example_rhs():
    # On x>=0, solve -u''-100u=sin(13*pi*x), with u'(0)=0,u(1)=0.
    # Even continuation gives the source example's sin(13*pi*abs(x)) forcing.
    k = 13*jnp.pi
    den = k*k - 100
    domain = (-1., 0., 1.)
    op = Chebop(lambda u: -u.diff(2) - 100*u, domain=domain)
    op.bc = 0
    f = cj.chebfun(lambda x: jnp.sin(k*jnp.abs(x)), domain=domain)
    exact = cj.chebfun(
        lambda x: (k*jnp.tan(10.)*jnp.cos(10*x)/10
                   - k*jnp.sin(10*jnp.abs(x))/10
                   + jnp.sin(k*jnp.abs(x))) / den, domain=domain)
    actual = krylov.minres(op, f, tol=5e-13, maxit=40)
    assert float((actual - exact).norm()) < SOURCE_TOL
    assert float((op(actual) - f).norm()) / float(f.norm()) < RESIDUAL_TOL
    assert float(jnp.max(jnp.abs(actual(jnp.array([-1., 1.]))))) < SOURCE_TOL
    # Independent interface continuity checks for the analytic even solution.
    assert abs(float(actual.funs[0](jnp.array(0.))
                     - actual.funs[1](jnp.array(0.)))) < SOURCE_TOL


def test_minres_indefinite_nonzero_boundary_correction():
    op = Chebop(lambda u: -u.diff(2) - 100*u)
    op.lbc, op.rbc = -1., 1.
    exact = cj.chebfun(lambda x: x + (1-x*x)*jnp.sin(3*x))
    f = cj.chebfun(lambda x: (2-91*(1-x*x))*jnp.sin(3*x)
                   + 12*x*jnp.cos(3*x) - 100*x)
    actual = krylov.minres(op, f, tol=5e-13, maxit=40)
    assert float((actual-exact).norm()) < SOURCE_TOL
    assert float((op(actual)-f).norm()) / float(f.norm()) < RESIDUAL_TOL
    assert float(jnp.max(jnp.abs(actual(jnp.array([-1., 1.]))
                                  - jnp.array([-1., 1.])))) < SOURCE_TOL


def test_minres_zero_rhs_returns_jax_residual_vector():
    op = Chebop(lambda u: -u.diff(2))
    op.bc = 0
    f = cj.chebfun(lambda x: 0*x)
    actual, flag, _, iteration, residuals = krylov.minres(op, f, full_output=True)
    assert float(actual.norm()) == 0
    assert flag == 0 and iteration == 0
    assert isinstance(residuals, jax.Array)
    assert residuals.shape == (1,) and float(residuals[0]) == 0
