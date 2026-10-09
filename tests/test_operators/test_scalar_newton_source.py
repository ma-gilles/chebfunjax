"""Scalar source Newton controls, Chebfun 7574c77 fitBCs/solvebvpNonlinear."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.operators.scalar_newton import _norm, fit_scalar_bcs


def test_gulf_callable_initial_guess():
    op = Chebop(lambda u: u.diff(3) + .1*(u.diff()**2-u*u.diff(2))-u+1,
                (0., 35.))
    op.lbc = lambda u: [u, u.diff(2)]
    op.rbc = 1.
    u = fit_scalar_bcs(op)
    xx = jnp.linspace(0, 35, 101)
    assert float(jnp.max(jnp.abs(u(xx)-xx/35))) < 30*jnp.finfo(float).eps


def test_numeric_derivatives_initial_guess():
    op = Chebop(lambda u: u.diff(3)+u*u, (0., 2.))
    op.lbc = [2., 3.]
    op.rbc = 16.
    u = fit_scalar_bcs(op)
    x = jnp.linspace(0, 2, 33)
    assert float(jnp.max(jnp.abs(u(x)-(2+3*x+2*x*x)))) < 100*jnp.finfo(float).eps


def test_trivial_second_derivative_constraint():
    op = Chebop(lambda u: u.diff(2)+u*u, (-1., 1.))
    op.lbc = lambda u: u.diff(2)-2
    # The first two grids have a trivial row, which source removes; source
    # therefore returns zero without pretending to satisfy this condition.
    assert float(fit_scalar_bcs(op).norm()) == 0


def test_general_boundary_conditions():
    op = Chebop(lambda u: u.diff(2)+u*u, (-1., 1.))
    op.bc = lambda u: [u(-1)-2, u(1)-4]
    u = fit_scalar_bcs(op)
    assert float((u-chebfun(lambda x: 3+x)).norm()) < 100*jnp.finfo(float).eps


def test_continuous_norm_not_grid_norm():
    u = chebfun(lambda x: x, domain=(0., 3.))
    assert abs(_norm([u])-3.) < 10*jnp.finfo(float).eps


def test_explicit_initial_exact_solution():
    op = Chebop(lambda u: u.diff(2)+u*u, (-1., 1.), 1., 1.)
    op.init = chebfun(1.)
    u, info = op.solvebvp(1.)
    assert float((u-1).norm()) < 1e-13
    assert len(info['normDelta']) == 1
    assert jnp.isnan(info['error'])


def test_source_domain_limit():
    op = Chebop(lambda u: u.diff(2)+u*u, (-1., 0., 1.))
    with pytest.raises(ValueError, match='one finite interval'):
        fit_scalar_bcs(op)


def test_rank_retry_distinct_derivative_constraints():
    op = Chebop(lambda u: u.diff(2)+u*u, (-1., 1.))
    op.lbc = lambda u: u.diff()-1
    op.rbc = lambda u: u.diff()-3
    u = fit_scalar_bcs(op)
    assert abs(float(u.diff()(-1))-1) < 100*jnp.finfo(float).eps
    assert abs(float(u.diff()(1))-3) < 100*jnp.finfo(float).eps


@pytest.mark.parametrize('higher_order', [False, True])
def test_original_fitbcs_scalar_constraints(higher_order):
    # Literal scalar columns 1/2 of tests/linop/test_fitBCs.m. The source
    # overrides all three requested discretizations with Chebcolloc2.
    op = Chebop(lambda u: u.diff(3 if higher_order else 2), (-1., 1.))
    op.bc = (lambda x, u: [u.sum()-2, u(-1)+u(1)-3, u.diff(2)(0)]) if higher_order else (
        lambda x, u: [u.sum()-2, u(-1)+u(1)-3])
    u = fit_scalar_bcs(op)
    residual = op.bc(chebfun(lambda x: x), u)
    assert float(jnp.linalg.norm(jnp.asarray(residual))) < 1e-10


def test_algebraic_fit_source_zero():
    op = Chebop(lambda u: u*u, (-1., 1.), 1., 1.)
    assert float(fit_scalar_bcs(op).norm()) == 0


def test_rank_retry_failure_warning():
    op = Chebop(lambda u: u.diff(2)+u*u, (-1., 1.))
    op.bc = lambda u: [u(0)-1, u(0)-2]
    with pytest.warns(RuntimeWarning, match='CHEBFUN:LINOP:fitBCs:failure'):
        u = fit_scalar_bcs(op)
    assert float(u.norm()) == 0
