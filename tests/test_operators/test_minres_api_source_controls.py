"""Unrun scratch controls for the public MINRES API/source edge contracts.

Provenance
----------
MATLAB source: @chebop/minres.m, @chebfun/normest.m,
               tests/chebop/test_minres.m
Chebfun source commit: 7574c77
These controls are not installed in the checkout and have not been run.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

BVP_TOL = 5e-13
MAX_ITER = 25
EPS = float(jnp.finfo(jnp.float64).eps)


def _op_laplacian():
    return Chebop(lambda x, u: -(u.diff()).diff(), domain=(-1.0, 1.0))


def _manufactured_mean_zero_case():
    # -u'' = 1 - 3x^2, u(+-1)=0, exact u=(1-x^2)^2/4.
    N = _op_laplacian()
    N.lbc = N.rbc = 0.0
    f = cj.chebfun(lambda x: 1.0 - 3.0*x**2, domain=(-1.0, 1.0))
    exact = cj.chebfun(lambda x: (1.0 - x**2)**2/4.0,
                       domain=(-1.0, 1.0))
    return N, f, exact


def _l2_error(actual, expected):
    return float((actual - expected).norm(2))


@pytest.mark.parametrize(
    "tol,maxit",
    [
        (None, None),
        ([], []),
        (jnp.asarray([], dtype=jnp.float64),
         jnp.asarray([], dtype=jnp.int32)),
        (None, []),
        ([], None),
    ],
    ids=("omitted-defaults", "matlab-empty-lists", "jax-empty-arrays",
         "empty-maxit", "empty-tol"),
)
def test_source_empty_default_arguments_solve_independent_manufactured_case(tol, maxit):
    N, f, exact = _manufactured_mean_zero_case()
    actual = N.minres(f, tol=tol, maxit=maxit)
    # Independent analytic solution; this does not compare empty-option calls
    # against another invocation of the same implementation.
    assert _l2_error(actual, exact) < 100.0 * BVP_TOL


def test_validation_checks_linearity_before_differential_order():
    f = cj.chebfun(lambda x: 1.0 + 0.0*x, domain=(-1.0, 1.0))
    nonlinear_first_order = Chebop(
        lambda x, u: u.diff() + u*u, domain=(-1.0, 1.0)
    )
    with pytest.raises(ValueError, match="CHEBFUN:CHEBOP:pcg:nonlinear"):
        nonlinear_first_order.minres(f)


def test_validation_rejects_linear_operator_not_second_order():
    f = cj.chebfun(lambda x: 1.0 + 0.0*x, domain=(-1.0, 1.0))
    first_order = Chebop(lambda x, u: u.diff() + u,
                         domain=(-1.0, 1.0))
    with pytest.raises(ValueError, match="CHEBFUN:CHEBOP:pcg:DiffOrder"):
        first_order.minres(f)


def test_validation_rejects_callable_dirichlet_boundary_data():
    f = cj.chebfun(lambda x: 1.0 + 0.0*x, domain=(-1.0, 1.0))
    N = _op_laplacian()
    N.lbc = lambda u: u
    with pytest.raises(ValueError, match="Currently, we require Dirichlet"):
        N.minres(f)


@pytest.mark.parametrize("which", ["R1", "R2"])
def test_validation_rejects_nondefault_preconditioner(which):
    N = _op_laplacian()
    f = cj.chebfun(lambda x: 1.0 - 3.0*x**2, domain=(-1.0, 1.0))
    kwargs = {which: lambda u: u}
    with pytest.raises(ValueError, match="OnlyDefaultPreconditionerAllowed"):
        N.minres(f, **kwargs)


def test_valid_zero_initial_guess_and_empty_initial_guess_forms():
    N = _op_laplacian()
    f = cj.chebfun(lambda x: 0.0*x, domain=(-1.0, 1.0))
    expected = cj.chebfun(lambda x: 0.0*x, domain=(-1.0, 1.0))
    for u0 in (None, [], jnp.asarray([], dtype=jnp.float64), expected):
        actual, flag, relres, iteration, resvec = N.minres(
            f, u0=u0, full_output=True
        )
        assert _l2_error(actual, expected) == 0.0
        assert flag == 0
        assert iteration == 0
        assert len(resvec) == 1 and float(resvec[0]) == 0.0
        # Source branch computes 0/0 when f=0; this is preserved and labeled.
        assert jnp.isnan(relres)



@pytest.mark.parametrize(
    "R1,R2",
    [
        (None, None),
        ([], []),
        (jnp.asarray([], dtype=jnp.float64),
         jnp.asarray([], dtype=jnp.float64)),
    ],
    ids=("omitted", "matlab-empty", "jax-empty-adapter"),
)
def test_empty_preconditioner_arguments_select_source_default(R1, R2):
    N, f, exact = _manufactured_mean_zero_case()
    actual = N.minres(f, tol=BVP_TOL, maxit=MAX_ITER, R1=R1, R2=R2)
    assert _l2_error(actual, exact) < 100.0 * BVP_TOL

def test_initial_guess_with_different_domain_raises_source_error():
    N, f, _ = _manufactured_mean_zero_case()
    wrong_domain = cj.chebfun(lambda x: 0.0*x, domain=(-0.9, 1.0))
    with pytest.raises(ValueError, match="WrongInitGuessDomain"):
        N.minres(f, u0=wrong_domain)


def _source_r2(v):
    # Literal source adjoint of cumsum: sum(v)-cumsum(v).
    return v.sum() - v.cumsum()


def _source_pi(v):
    # Scalar mean is the continuous integral divided by domain length.
    return v - v.mean()


def test_full_output_relative_residual_uses_continuous_l2_rhs_norm():
    # For -u'' + u = 1 - 3x^2, the exact Dirichlet solution is
    # -5 - 3x^2 + 8*cosh(x)/cosh(1). The independent exact-polynomial
    # Lanczos oracle below predicts a nonzero first-step residual.
    N = Chebop(lambda x, u: -(u.diff()).diff() + u,
               domain=(-1.0, 1.0))
    N.lbc = N.rbc = 0.0
    # This zero-mean RHS makes source Pi the identity on R2(f), hence z=0.
    f = cj.chebfun(lambda x: 1.0 - 3.0*x**2, domain=(-1.0, 1.0))
    u, flag, relres, iteration, resvec = N.minres(
        f, tol=0.5, maxit=2, full_output=True
    )
    direct_projected_residual = _source_pi(
        _source_r2(f - N.feval(u))
    )
    expected_relres = float(direct_projected_residual.norm(2) / f.norm(2))
    # Independent Fraction-integration oracle for source MINRES iteration 1:
    # <g,g>=16/105, <g,Tg>=64/315, <Tg,Tg>=14248/51975, giving
    # residual^2=16/8905 and relative-residual^2=2/1781.
    expected_first_relres = jnp.sqrt(2.0/1781.0)
    assert expected_relres > 100.0*EPS
    assert abs(expected_relres - float(expected_first_relres)) <= 100.0*EPS
    assert abs(float(relres) - expected_relres) <= 100.0*EPS*max(1.0, expected_relres)
    assert flag == 0 and iteration == 1
    assert len(resvec) >= 1


def test_initial_relative_tolerance_uses_sum_of_piece_normestimates():
    # Let g be the centered triangular function with break at zero.  Its two
    # linear pieces each have source normest 1/2, so normest(g)=1 by
    # @chebfun/normest.m's piecewise sum.  Its continuous L2 norm is sqrt(1/6).
    # For this f, R2(f)-Pi(R2(f)) has L2 norm sqrt(1/2), below .75, so
    # the source range-correction z remains zero.  tol=.75 then makes the
    # initial residual test pass with the piece sum (.75*1) but fail if an
    # implementation mistakenly uses only max(.5).
    N = Chebop(lambda x, u: -(u.diff()).diff() + u,
               domain=(-1.0, 0.0, 1.0))
    N.lbc = N.rbc = 0.0
    # Construct the two one-sided constants explicitly. A single step callable
    # samples +1 at the left panel's right endpoint, creating an unhappy
    # approximation instead of the exact piecewise L2 input used by this oracle.
    f = cj.chebfun([lambda x: -jnp.ones_like(x), jnp.ones_like],
                   domain=(-1.0, 0.0, 1.0))
    assert all(piece.tech.ishappy for piece in f.funs)
    assert abs(float(f.norm(2)) - 2.0**0.5) <= 100.0*EPS
    u, flag, relres, iteration, resvec = N.minres(
        f, tol=0.75, maxit=MAX_ITER, full_output=True
    )
    assert flag == 0 and iteration == 0
    assert abs(float(resvec[0]) - (1.0/6.0)**0.5) <= 100.0*EPS
    assert abs(float(relres) - (1.0/12.0)**0.5) <= 100.0*EPS


def test_source_error_precedence_nonlinearity_before_order_bc_tol_and_preconditioner():
    f = cj.chebfun(lambda x: 1.0 + 0.0*x, domain=(-1.0, 1.0))
    N = Chebop(lambda x, u: u.diff() + u*u, domain=(-1.0, 1.0))
    N.lbc = lambda u: u
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="CHEBFUN:CHEBOP:pcg:nonlinear"):
            N.minres(f, tol=1.5, R1=lambda u: u)
    assert not seen  # MATLAB rejects nonlinearity before tolerance clamping.


def test_source_error_precedence_order_before_boundary_tolerance_and_preconditioner():
    f = cj.chebfun(lambda x: 1.0 + 0.0*x, domain=(-1.0, 1.0))
    N = Chebop(lambda x, u: u.diff() + u, domain=(-1.0, 1.0))
    N.lbc = lambda u: u
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="CHEBFUN:CHEBOP:pcg:DiffOrder"):
            N.minres(f, tol=1.5, R1=lambda u: u)
    assert not seen  # MATLAB rejects differential order before tolerance handling.


def test_source_error_precedence_boundary_before_tolerance_and_preconditioner():
    f = cj.chebfun(lambda x: 1.0 + 0.0*x, domain=(-1.0, 1.0))
    N = _op_laplacian()
    N.lbc = lambda u: u
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="Currently, we require Dirichlet"):
            N.minres(f, tol=1.5, R1=lambda u: u)
    assert not seen  # MATLAB checks numeric BCs before tolerance/preconditioner.
