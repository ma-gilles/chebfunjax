"""Independent controls for exact AD blocks and source adaptive dimensions."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators._linear_altdisc import (
    LinearDiscretization,
    _dimension_values,
)
from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import D, I, eval_at, zero_functional
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize('backend', ['chebcolloc1', 'ultraS'])
def test_linear_route_uses_exact_ad_without_perturbations(monkeypatch, backend):
    import chebfunjax.operators.chebop_altdisc as module

    def forbidden(*args, **kwargs):
        raise AssertionError('Coupled linear route must not perturb the operator')

    monkeypatch.setattr(module, '_frechet_blocks', forbidden)
    op = Chebop(lambda x, u, v: [u.diff()-v, v.diff()], (-1., 1.))
    op.lbc = lambda u, v: [u+1, v-1]
    u, v = op.solve(0, n=9, discretization=backend)
    x = chebfun(lambda t: t)
    assert (u-x).norm(jnp.inf) < 1e-10
    assert (v-1).norm(jnp.inf) < 1e-10


@pytest.mark.parametrize('backend', ['chebcolloc1', 'ultraS'])
def test_source_input_dimensions_include_column_orders(backend):
    dom = (-1., 1.)
    derivative, identity = D(dom), I(dom)
    operator = linop(ChebMatrix([[derivative**2, identity],
                                [identity, derivative]]))
    operator = operator.add_constraint([eval_at(-1., dom), zero_functional(dom)], 0.)
    operator = operator.add_constraint([eval_at(1., dom), zero_functional(dom)], 0.)
    operator = operator.add_constraint([zero_functional(dom), eval_at(-1., dom)], 0.)
    disc = LinearDiscretization(operator, (8,), backend)
    assert disc.col_dimensions == (10, 9)
    assert disc.A.shape == (19, 19)
    assert [p.shape for p in disc.projections] == [(8, 10), (8, 9)]
    disc.solve(disc.rhs([0., 0.]))
    # Native valsDiscretization.isFactored compares against equation size16,
    # so these19-row stored LU factors are not considered reusable.
    assert not disc.is_factored


def test_source_non_power_two_dimension_schedule():
    assert _dimension_values(300, 2048, 'chebcolloc1') == (300, 724, 1024, 1448, 2048)
    assert _dimension_values(300, 2048, 'ultraS') == (300, 600, 1200)


def test_autonomous_cell_constructor_normalizes_system_arity():
    op = Chebop(lambda u: [u[0].diff()+u[1], u[1].diff()], (-1., 1.))
    assert op._n_vars() == 2


def test_ad_preserves_fifth_derivative_and_small_coupling():
    from chebfunjax.operators.chebop_altdisc import _linearize_coupled_ad

    op = Chebop(lambda x, u, v: [u.diff(5)+1e-9*v.diff(2), v.diff()], (-1., 1.))
    op.lbc = lambda u, v: u.diff(4)+1e-9*v
    operator, _rhs, _initial = _linearize_coupled_ad(op, [0., 0.], (-1., 1.), 2)
    x = chebfun(lambda t: t)
    assert operator.A.blocks[0][0].order == 5
    assert (operator.A.blocks[0][0].apply(x**5)-120).norm(jnp.inf) < 1e-12
    assert (operator.A.blocks[0][1].apply(x**2)-2e-9).norm(jnp.inf) < 1e-21
    boundary, _value = operator.constraint[0]
    assert jnp.abs(boundary[0].apply(x**4)-24) < 1e-12
    assert jnp.abs(boundary[1].apply(chebfun(1.))-1e-9) < 1e-21
