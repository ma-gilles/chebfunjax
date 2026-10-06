"""Source GMRES options, exact restart oracles and independent branch controls.

Provenance
----------
MATLAB source : @chebop/gmres.m, tests/chebop/test_gmres.m
Chebfun commit: 7574c77

These controls preserve source residual denominators and strict polynomial
oracles. The first source early exit leaves ITER unassigned; the Python
multiple-return adapter uses [0,0], explicitly tested as an adapter.
"""
from __future__ import annotations

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop
from chebfunjax.operators.krylov import gmres

_PREF = ChebopPref()
_TOL = _PREF.bvpTol
_MAXIT = _PREF.maxIter
_COMPARE_BOUND = 1.0e2 * _TOL


def _problem():
    n = Chebop(lambda x, u: -u.diff(2) + u, domain=(-1.0, 1.0))
    n.bc = 0.0
    f = cj.chebfun(lambda x: 1.0 - 3.0 * x**2, domain=(-1.0, 1.0))
    return n, f


def _l2diff(u, v):
    return float((u - v).norm(2))


def test_default_empty_and_explicit_options_match_factory():
    n, f = _problem()
    default = gmres(n, f, full_output=True)
    empty = gmres(n, f, restart=None, tol=None, maxit=None, R1=None,
                  R2=None, u0=None, full_output=True)
    explicit = gmres(n, f, restart=None, tol=_TOL, maxit=_MAXIT,
                     R1=None, R2=None, u0=None, full_output=True)
    assert empty[1] == explicit[1] == default[1]
    assert jnp.array_equal(jnp.asarray(empty[3]), jnp.asarray(explicit[3]))
    assert _l2diff(empty[0], explicit[0]) < _COMPARE_BOUND
    assert _l2diff(default[0], empty[0]) < _COMPARE_BOUND


def test_restart_one_per_outer_cycle_matches_two_step_oracle():
    n, f = _problem()
    # Exact Fraction-polynomial recurrence is recorded in the companion
    # oracle JSON: alpha_1=1320/1781, r_1=g-alpha_1*T(g), then
    # alpha_2=117505080/128572561 and v_2=alpha_1*g+alpha_2*r_1.
    # ||r_1||/||g||=sqrt(21/1781)>.05, while
    # ||r_2||/||g||=sqrt(748750370613/5301752939107573)<.05.
    # Thus source GMRES with restart=1 and maxit=2 must take exactly two
    # outer cycles and stop on inner iteration one of the second cycle.
    u, flag, relres, iteration, residuals = gmres(
        n, f, restart=1, tol=0.05, maxit=2, full_output=True
    )
    den = 228987731141.0
    expected = cj.chebfun(
        lambda x: (41753290920.0 - 92554473000.0 * x**2
                   + 49508626200.0 * x**4 + 1292555880.0 * x**6) / den,
        domain=(-1.0, 1.0),
    )
    eps = jnp.finfo(jnp.float64).eps
    assert flag == 0
    assert tuple(int(v) for v in jnp.asarray(iteration)) == (2, 1)
    assert int(jnp.asarray(residuals).size) == 3
    assert _l2diff(u, expected) < 300.0 * eps
    expected_relres = jnp.sqrt(748750370613.0 / 5301752939107573.0)
    assert abs(float(relres) - float(expected_relres)) < 300.0 * eps
    expected_f_rel = jnp.sqrt(71309559106.0 / 5301752939107573.0)
    assert abs(float(residuals[-1] / f.norm(2))
               - float(expected_f_rel)) < 300.0 * eps


def test_one_unrestarted_step_matches_independent_polynomial_oracle():
    n, f = _problem()
    # Source R1(v)=integral[-1,x]v and R2(v)=integral[x,1]v give
    # g=Pi(R2 f)=x^3-x and T(g)=Pi(R2 L R1 g)=
    # -x^5/20+7x^3/6-5x/4. Exact polynomial inner products yield the
    # one-step coefficient alpha=1320/1781 and returned R1(alpha*g)
    # =330/1781*(1-x^2)^2.
    u, flag, relres, iteration, residuals = gmres(
        n, f, restart=None, tol=0.5, maxit=1, full_output=True
    )
    expected = cj.chebfun(
        lambda x: (330.0 / 1781.0) * (1.0 - x**2) ** 2,
        domain=(-1.0, 1.0),
    )
    eps = jnp.finfo(jnp.float64).eps
    assert flag == 0
    assert tuple(int(v) for v in jnp.asarray(iteration)) == (1, 1)
    assert _l2diff(u, expected) < 100.0 * eps
    # relres is relative to ||g||; the stored transformed residual divided
    # by ||f|| is sqrt(2/1781), per the source residual construction.
    assert abs(float(relres) - float(jnp.sqrt(21.0 / 1781.0))) < 100.0 * eps
    assert abs(float(residuals[-1] / f.norm(2))
               - float(jnp.sqrt(2.0 / 1781.0))) < 100.0 * eps


def test_reported_solution_has_independent_physical_residual():
    n, f = _problem()
    u, flag, relres, _, _ = gmres(
        n, f, restart=None, tol=_TOL, maxit=_MAXIT, full_output=True
    )
    exact = cj.chebfun(
        lambda x: -5.0 - 3.0 * x**2 + 8.0 * jnp.cosh(x) / jnp.cosh(1.0),
        domain=(-1.0, 1.0),
    )
    physical_residual = (n.feval(u) - f).norm(2) / f.norm(2)
    assert flag == 0
    assert float(physical_residual) < _COMPARE_BOUND
    assert _l2diff(u, exact) < _COMPARE_BOUND
    assert jnp.isfinite(relres)


def test_nonempty_r1_is_rejected():
    n, f = _problem()
    with pytest.raises(ValueError, match="OnlyDefaultPreconditionerAllowed"):
        gmres(n, f, restart=None, tol=_TOL, maxit=_MAXIT,
              R1=lambda v: v, R2=None)


def test_nonempty_r2_is_also_rejected_by_literal_source():
    n, f = _problem()
    with pytest.raises(ValueError, match="OnlyDefaultPreconditionerAllowed"):
        gmres(n, f, restart=None, tol=_TOL, maxit=_MAXIT,
              R1=None, R2=lambda v: v)


def test_initial_guess_domain_mismatch_is_rejected():
    n, f = _problem()
    u0 = cj.chebfun(lambda x: 0.0 * x, domain=(0.0, 1.0))
    with pytest.raises(ValueError, match="WrongInitGuessDomain"):
        gmres(n, f, restart=None, tol=_TOL, maxit=_MAXIT,
              R1=None, R2=None, u0=u0)


def test_non_dirichlet_boundary_is_rejected():
    n, f = _problem()
    n.lbc = lambda u: u.diff()
    with pytest.raises(ValueError, match="leftbc"):
        gmres(n, f)


def test_nonlinear_operator_is_rejected_before_iteration():
    n = Chebop(lambda x, u: -u.diff(2) + u**2, domain=(-1.0, 1.0))
    n.bc = 0.0
    f = cj.chebfun(lambda x: 1.0 - 3.0 * x**2)
    with pytest.raises(ValueError, match="GMRES supports only linear"):
        gmres(n, f)


def test_non_second_order_operator_is_rejected():
    n = Chebop(lambda x, u: u.diff(), domain=(-1.0, 1.0))
    n.bc = 0.0
    f = cj.chebfun(lambda x: x, domain=(-1.0, 1.0))
    with pytest.raises(ValueError, match="GMRES supports only second-order"):
        gmres(n, f)


def test_tolerance_warning_precedes_preconditioner_rejection():
    n, f = _problem()
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError, match="OnlyDefaultPreconditionerAllowed"):
            gmres(n, f, restart=None, tol=0.0, maxit=None,
                  R1=lambda v: v, R2=None)


def test_homogeneous_zero_rhs_has_nan_relres_and_stable_iteration_adapter():
    n, _ = _problem()
    f = cj.chebfun(0.0, domain=(-1.0, 1.0))
    u, flag, relres, iteration, residuals = gmres(n, f, full_output=True)
    # MATLAB first return supplies norm(r)/norm(f)=0/0 but does not assign
    # ITER. The Python multiple-return adapter documents [0,0] in that slot.
    assert flag == 0
    assert float(u.norm(2)) == 0.0
    assert jnp.isnan(relres)
    assert tuple(int(v) for v in jnp.asarray(iteration)) == (0, 0)
    assert jnp.asarray(residuals).shape == (1,)
    assert float(residuals[0]) == 0.0


def test_one_argument_callback_matches_independent_cosh_solution():
    n = Chebop(lambda u: -u.diff(2) + u, domain=(-1.0, 1.0))
    n.bc = 0.0
    f = cj.chebfun(lambda x: 1.0 - 3.0 * x**2)
    u = gmres(n, f)
    exact = cj.chebfun(
        lambda x: -5.0 - 3.0 * x**2 + 8.0 * jnp.cosh(x) / jnp.cosh(1.0)
    )
    assert _l2diff(u, exact) < _COMPARE_BOUND
    assert float((n.op(u) - f).norm(2) / f.norm(2)) < _COMPARE_BOUND


def test_two_argument_coefficient_mining_uses_rhs_domain():
    # Source validates linop(N) on N.domain but builds mining x on domain(f).
    # u=x(1-x), L=-d^2+x, f=2+x^2(1-x) on [0,1] is independent analytic data.
    n = Chebop(lambda x, u: -u.diff(2) + x * u, domain=(-1.0, 1.0))
    n.bc = 0.0
    f = cj.chebfun(lambda x: 2.0 + x**2 * (1.0 - x), domain=(0.0, 1.0))
    u = gmres(n, f)
    exact = cj.chebfun(lambda x: x * (1.0 - x), domain=(0.0, 1.0))
    x = cj.chebfun(lambda x: x, domain=(0.0, 1.0))
    assert u.domain.breakpoints == f.domain.breakpoints
    assert _l2diff(u, exact) < _COMPARE_BOUND
    assert float((n.op(x, u) - f).norm(2) / f.norm(2)) < _COMPARE_BOUND
