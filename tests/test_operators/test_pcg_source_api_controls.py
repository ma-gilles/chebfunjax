"""Source API controls for MATLAB @chebop/pcg.m.

Provenance
----------
MATLAB source : tests/chebop/test_pcg.m, @chebop/pcg.m,
                @cheboppref/cheboppref.m
Chebfun commit: 7574c77

These bounded controls use `None` for MATLAB's empty optional tolerance,
iteration, preconditioner, and initial-guess slots. Exact analytic oracles
check physical and transformed residuals separately. Captured CPU execution
and source comparison are recorded in the separate qualification evidence.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop
from chebfunjax.operators.krylov import pcg

_PREF = ChebopPref()
_SOURCE_TOL = _PREF.bvpTol
_SOURCE_MAXIT = _PREF.maxIter
_TOL = 1e2 * _SOURCE_TOL


def _constant_problem(domain=(-1.0, 1.0)):
    n = Chebop(lambda x, u: -u.diff(2) + u, domain=domain)
    n.bc = 0.0
    f = cj.chebfun(lambda x: 1.0 - 3.0 * x**2, domain=domain)
    return n, f


def _error_norm(u, v):
    return float((u - v).norm(2))


def test_source_default_options_match_empty_option_slots():
    n, f = _constant_problem()
    implicit = pcg(n, f, full_output=True)
    empty_slots = pcg(n, f, tol=None, maxit=None, R1=None, R2=None,
                      u0=None, full_output=True)
    explicit_factory = pcg(n, f, tol=_SOURCE_TOL, maxit=_SOURCE_MAXIT,
                           R1=None, R2=None, u0=None, full_output=True)
    assert empty_slots[1] == explicit_factory[1]
    assert empty_slots[3] == explicit_factory[3]
    assert _error_norm(implicit[0], empty_slots[0]) < _TOL
    assert _error_norm(empty_slots[0], explicit_factory[0]) < _TOL


def test_source_zero_rhs_early_return_nan_relative_residual():
    n, _ = _constant_problem()
    f = cj.chebfun(lambda x: 0.0 * x, domain=(-1.0, 1.0))
    u, flag, relres, iteration, resvec = pcg(
        n, f, tol=None, maxit=None, full_output=True
    )
    assert flag == 0
    assert iteration == 0
    assert float(u.norm(2)) == 0.0
    assert jnp.isnan(relres)
    assert jnp.asarray(resvec).shape == (1,)
    assert float(resvec[0]) == 0.0


def test_source_returned_solution_has_independent_physical_residual():
    n, f = _constant_problem()
    u, flag, relres, _, _ = pcg(
        n, f, tol=_SOURCE_TOL, maxit=_SOURCE_MAXIT, full_output=True
    )
    residual = n.feval(u) - f
    physical_relres = float(residual.norm(2) / f.norm(2))
    exact = cj.chebfun(
        lambda x: -5.0 - 3.0 * x**2 + 8.0 * jnp.cosh(x) / jnp.cosh(1.0)
    )
    assert flag == 0
    assert physical_relres < _TOL
    assert _error_norm(u, exact) < _TOL
    assert jnp.isfinite(relres)


def test_source_one_step_residual_and_solution_exact_polynomial_oracle():
    n, f = _constant_problem()
    # Independent hand derivation for L=-u''+u: g=x^3-x, T(g)=
    # -x^5/20+7x^3/6-5x/4, so alpha=3/4 and ||r_1||^2=1/550.
    # Since ||g||^2=16/105 and tol=.5, source PCG stops after this step.
    u, flag, relres, iteration, resvec = pcg(
        n, f, tol=0.5, maxit=1, full_output=True
    )
    expected = cj.chebfun(lambda x: (3.0 / 16.0) * (1.0 - x**2)**2)
    eps = jnp.finfo(jnp.float64).eps
    assert flag == 0
    assert iteration == 1
    assert _error_norm(u, expected) < 100.0 * eps
    assert abs(float(resvec[-1]) - float(jnp.sqrt(1.0 / 550.0))) < 100.0 * eps
    assert abs(float(relres) - float(jnp.sqrt(1.0 / 880.0))) < 100.0 * eps


def test_source_nonempty_r1_is_rejected():
    n, f = _constant_problem()
    with pytest.raises(ValueError, match="OnlyDefaultPreconditionerAllowed"):
        pcg(n, f, tol=None, maxit=None, R1=lambda v: v, R2=None)


def test_source_custom_r2_is_allowed_with_default_r1():
    n, f = _constant_problem()

    def source_adjoint(v):
        return float(v.sum()) - v.cumsum()

    baseline = pcg(n, f)
    custom_r2 = pcg(n, f, tol=None, maxit=None, R1=None,
                    R2=source_adjoint)
    assert _error_norm(baseline, custom_r2) < _TOL


def test_source_initial_guess_domain_mismatch_is_rejected():
    n, f = _constant_problem()
    wrong_domain_guess = cj.chebfun(
        lambda x: 0.0 * x, domain=(0.0, 1.0)
    )
    with pytest.raises(ValueError, match="WrongInitGuessDomain"):
        pcg(n, f, tol=None, maxit=None, R1=None, R2=None,
            u0=wrong_domain_guess)


def test_source_non_dirichlet_boundary_is_rejected():
    n, f = _constant_problem()
    n.lbc = lambda u: u.diff()
    with pytest.raises(ValueError, match="Dirichlet boundary conditions"):
        pcg(n, f)


def test_source_nonlinear_operator_is_rejected_before_iteration():
    n = Chebop(lambda x, u: -u.diff(2) + u**2, domain=(-1.0, 1.0))
    n.bc = 0.0
    f = cj.chebfun(lambda x: 1.0 - 3.0 * x**2)
    with pytest.raises(ValueError, match="PCG supports only linear"):
        pcg(n, f)


def test_source_non_second_order_operator_is_rejected():
    n = Chebop(lambda x, u: u.diff(), domain=(-1.0, 1.0))
    n.bc = 0.0
    f = cj.chebfun(lambda x: x, domain=(-1.0, 1.0))
    with pytest.raises(ValueError, match="PCG supports only second-order"):
        pcg(n, f)


def test_source_tolerance_warning_precedes_preconditioner_error():
    n, f = _constant_problem()
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError, match="OnlyDefaultPreconditionerAllowed"):
            pcg(n, f, tol=0.0, maxit=None, R1=lambda v: v)
