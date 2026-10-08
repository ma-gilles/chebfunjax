"""Analytic sum controls for orientation, point values and periodic data."""

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix


def test_reversed_valid_limits_and_zero_width():
    f = cj.chebfun(lambda x: x*x, domain=(-2, 0, 3))
    assert jnp.abs(f.sum(2, -1)+3) < 2e-15
    assert f.sum(.5, .5) == 0


def test_discrete_sum_preserves_point_values_and_orientation():
    f = cj.chebfun(lambda x: jnp.stack([x, x*x], axis=-1))
    object.__setattr__(f, '_point_values', jnp.array([[3., 4.], [5., 6.]]))
    for g in (f.sum(2), f.T.sum(1)):
        assert jnp.array_equal(g(jnp.array([-1., 1.])), jnp.array([7., 11.]))
    assert f.T.sum(1).is_transposed
    assert f.T.sum().shape == (2, 1)


def test_periodic_mean_and_subinterval():
    f = cj.chebfun(lambda x: 2+jnp.sin(x), domain=(-jnp.pi, jnp.pi), trig=True)
    assert jnp.abs(f.sum()-4*jnp.pi) < 2e-14
    assert jnp.abs(f.sum(0., jnp.pi)-(2*jnp.pi+2)) < 2e-14
    assert jnp.abs(f.T.sum()-4*jnp.pi) < 2e-14


def test_quasimatrix_dimensions_and_functional_limits():
    x = cj.chebfun(lambda x: x)
    q = Quasimatrix([x, x*x], x.domain)
    assert jnp.max(jnp.abs(q.sum(2)(jnp.array([-.5, .3]))-jnp.array([-.25, .39]))) < 2e-15
    h = q.sum(0., x)
    t = jnp.array([-.5, .3])
    assert jnp.max(jnp.abs(h(t)-jnp.stack([t*t/2, t**3/3], axis=-1))) < 2e-15


def test_unbounded_small_nonzero_endpoint_is_divergent():
    f = cj.chebfun(lambda x: jnp.exp(-x)+1e-9, domain=(0, jnp.inf))
    assert jnp.isposinf(f.sum())


@pytest.mark.parametrize('row,dim,empty_result', [(False, 1, False), (False, 2, True),
                                                (True, 1, True), (True, 2, False)])
def test_empty_dimension_policy(row, dim, empty_result):
    f = cj.chebfun()
    if row:
        f = f.T
    result = f.sum(dim)
    assert result.isempty() if empty_result else result == 0


def test_unbounded_string_identity_endpoint():
    x = cj.chebfun('x', domain=(0, jnp.inf))
    assert jnp.isposinf(x(jnp.inf))
    points = jnp.array([0., .1, 1., 20.])
    assert jnp.max(jnp.abs(x(points)-points)) < 3e-14


@pytest.mark.parametrize('side', ['left', 'right'])
def test_singular_half_line_gamma_integral(side):
    if side == 'right':
        f = cj.chebfun(lambda x: jnp.sqrt(x)*jnp.exp(-x),
                       domain=(0, jnp.inf), exps=(.5, 0))
    else:
        f = cj.chebfun(lambda x: jnp.sqrt(-x)*jnp.exp(x),
                       domain=(-jnp.inf, 0), exps=(0, .5))
    assert jnp.abs(f.sum()-jnp.sqrt(jnp.pi)/2) < 1e-12


@pytest.mark.parametrize('side', ['left', 'right'])
def test_cancelled_singular_product_demotes_before_integration(side):
    from chebfunjax.fun.singfun import Singfun
    from chebfunjax.tech.chebtech import Chebtech2

    exps = (-1., 0.) if side == 'left' else (0., -1.)
    pole = Singfun(Chebtech2.from_coeffs(jnp.array([1.])), exps)
    root = Chebtech2.from_coeffs(jnp.array([1., 1. if side == 'left' else -1.]))
    product = pole*root
    assert isinstance(product, Chebtech2)
    assert jnp.abs(product.sum()-2.) < 2e-15
