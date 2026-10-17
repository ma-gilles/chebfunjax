"""Scalar Newton corrections retain coefficient-induced interval boundaries.

Provenance: Chebfun7574c77 @linop/linsolve.m and chebcolloc2 reduction.
Independent polynomial solutions test the scalar dispatch boundary; these are
not native Bloodhound convergence or whole piecewise-Newton qualification.
"""
import jax.numpy as jnp
import pytest

from chebfunjax import chebfun
from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize('general_bc', [False, True])
def test_piecewise_coefficient_uses_interval_aware_correction(general_bc):
    coefficient = chebfun([2., 3.], domain=[0., .375, 1.])
    x = chebfun(lambda x: x, domain=[0., 1.])
    exact = x + 1
    forcing = coefficient + exact**2
    op = Chebop(lambda x, u: coefficient*u.diff()+u**2-forcing,
                domain=[0., 1.])
    if general_bc:
        op.bc = lambda x, u: u(0)-1
    else:
        op.lbc = 1.
    op.init = chebfun(0., domain=[0., 1.])
    seed = ADChebfun(op.init).seed(1, (True,))
    linearized = op._call_op(x, [seed])
    assert tuple(op.domain) == (0., 1.)
    assert tuple(linearized.jacobian.domain) == (0., .375, 1.)
    actual = op.solve(0., n=16, n_min=8, n_max=32)
    points = jnp.linspace(0., 1., 41)
    assert jnp.max(jnp.abs(actual(points).reshape(-1)-(points+1))) < 1e-10
    assert abs(actual(0.).item()-1.) < 1e-12
    assert any(abs(float(b)-.375) < 1e-15 for b in actual.domain.breakpoints)


def test_smooth_coefficient_keeps_existing_fast_correction(monkeypatch):
    from chebfunjax.operators import _linear_altdisc

    def unexpected(*args, **kwargs):
        raise AssertionError('single-interval correction unexpectedly delegated')

    monkeypatch.setattr(_linear_altdisc, 'solve_operator', unexpected)
    op = Chebop(lambda x, u: 2*u.diff()+u**2-2-(x+1)**2,
                domain=[0., 1.])
    op.lbc = 1.
    op.init = chebfun(0., domain=[0., 1.])
    actual = op.solve(0., n=16, n_min=8, n_max=32)
    points = jnp.linspace(0., 1., 41)
    assert jnp.max(jnp.abs(actual(points).reshape(-1)-(points+1))) < 1e-10


def test_nearby_coefficient_breaks_use_merged_equation_points():
    # @domain/merge coalesces these RHS breaks. Native vals RHS evaluates
    # equationPoints, so a tiny intervening FUN need not be restricted away.
    from chebfunjax.operators._second_kind_linear import SecondKindDiscretization
    from chebfunjax.operators.blocklinop import BlockLinop
    from chebfunjax.operators.blocks import D, eval_at
    from chebfunjax.utils.quadrature import chebpts

    domain = (0., .375, 1.)
    operator = BlockLinop(D(domain, 1), domain=domain)
    operator = operator.add_constraint(eval_at(0., domain), 7.)
    disc = SecondKindDiscretization(operator, (8, 8))
    residual = chebfun([1., 99., 2.], domain=[0., .375, .375+1e-15, 1.])
    assert len(residual.funs) == 3
    points = jnp.concatenate([
        (b*(chebpts(8, kind=1)+1)+a*(1-chebpts(8, kind=1)))/2
        for a, b in zip(domain[:-1], domain[1:])])
    values = disc.rhs([residual])
    assert disc.continuity_count == 1
    assert jnp.array_equal(values[:2], jnp.array([0., 7.]))
    assert jnp.array_equal(values[2:], residual(points).reshape(-1))
    assert jnp.array_equal(disc.rhs([2.]), jnp.concatenate([
        jnp.array([0., 7.]), jnp.full((16,), 2.)]))
    assert jnp.array_equal(disc.rhs([lambda x: 2.]), disc.rhs([2.]))
    update = disc.recover(disc.solve(values))[0]
    expected = 7+points+jnp.maximum(points-.375, 0)
    assert jnp.max(jnp.abs(update(points).reshape(-1)-expected)) < 1e-12
