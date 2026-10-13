"""Boundary-root cancellation controls from native FUN division semantics.

Provenance
----------
MATLAB source: @classicfun/rdivide.m, @fun/extractBoundaryRoots.m,
    @singfun/rdivide.m, @chebfun/rdivide.m.
Chebfun commit: 7574c77.
These analytic controls use 10eps, separately from original test predicates.
"""
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.fun.singfun import Singfun

EPS = jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize('root', [0., -1., 1.])
@pytest.mark.parametrize('scale', [1., 1.+2j])
def test_removable_linear_root_has_finite_point_value(root, scale):
    denominator = cj.chebfun(lambda x: x-root)
    numerator = scale*denominator
    h = numerator/denominator
    assert all(not isinstance(p.tech,Singfun) for p in h.funs)
    points = jnp.asarray([-1., -.75, 0., .75, 1.])
    assert jnp.max(jnp.abs(h(points)-scale)) < 10*EPS*abs(scale)
    assert jnp.all(jnp.isfinite(h.point_values))


def test_true_pole_is_retained():
    x = cj.chebfun(lambda x: x)
    h = 1/x
    assert all(isinstance(p.tech,Singfun) for p in h.funs)
    assert [p.tech.exponents for p in h.funs] == [(0.,-1.),(-1.,0.)]
    points = jnp.asarray([-.75,-.25,.25,.75])
    assert jnp.max(jnp.abs(h(points)-1/points)) < 10*EPS*4
    assert jnp.isinf(h(0.))


def test_chebtech1_removable_root_retains_selected_tech():
    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.domain import Domain
    from chebfunjax.tech.chebtech import Chebtech1

    x = Chebfun(funs=[_Piece(tech=Chebtech1.from_coeffs(jnp.asarray([0.,1.])),
                            interval=(-1.,1.))], domain=Domain((-1.,1.)))
    h = x/x
    assert all(isinstance(p.tech,Chebtech1) for p in h.funs)
    assert jnp.max(jnp.abs(h(jnp.asarray([-1.,0.,1.]))-1)) < 10*EPS


@pytest.mark.parametrize('numeric_numerator', [False, True])
def test_public_row_orientation_with_denominator_root(numeric_numerator):
    x = cj.chebfun(lambda x: x).T
    h = 1/x if numeric_numerator else x/x
    assert h.is_transposed
    points = jnp.asarray([-.75,.25,.75])[:,None]
    expected = (1/points if numeric_numerator else jnp.ones_like(points)).T
    assert h(points).shape == expected.shape
    assert jnp.max(jnp.abs(h(points)-expected)) < 10*EPS*4
