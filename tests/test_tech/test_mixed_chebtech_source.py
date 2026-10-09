"""Mixed C1/C2 arithmetic controls for source @chebtech/plus.m (7574c77).

The source accepts either subclass, keeps the left class, and computes the
zero-output threshold with each original operand's own vscale.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize('left,right', [(Chebtech1, Chebtech2),
                                      (Chebtech2, Chebtech1)])
@pytest.mark.parametrize('sign', [1, -1])
def test_mixed_add_subtract_unequal_lengths(left, right, sign):
    f = left.from_coeffs(jnp.array([1., 2., -3., .5]), ishappy=False)
    g = right.from_coeffs(jnp.array([-.25, 1.]), ishappy=True)
    actual = f+g if sign == 1 else f-g
    points = jnp.linspace(-1, 1, 33)
    expected = 1+2*points-3*(2*points**2-1)+.5*(4*points**3-3*points)
    expected += sign*(-.25+points)
    assert type(actual) is left
    assert not actual.ishappy
    assert float(jnp.max(jnp.abs(actual(points)-expected))) < 1e-13


def test_cancellation_uses_original_grid_vscales():
    # C1's two values have scale sqrt(.5), C2's scale is1. Converting the
    # right operand to the left grid would lower .2eps*max(vscale) and miss
    # source's exact zero-output branch for this3.5e-17 constant residual.
    f = Chebtech1.from_coeffs(jnp.array([3.5e-17, -1.]), ishappy=False)
    g = Chebtech2.from_coeffs(jnp.array([0., 1.]), ishappy=True)
    assert float(f.vscale) < float(g.vscale)
    result = f+g
    assert type(result) is Chebtech1
    assert not result.ishappy
    assert result.coeffs.shape == (1,)
    assert float(result.coeffs[0]) == 0.


@pytest.mark.parametrize('left,right', [(Chebtech1, Chebtech2),
                                      (Chebtech2, Chebtech1)])
def test_public_mixed_chebfun_point_values(left, right):
    domain = Domain((-1., 1.))
    f = Chebfun([_Piece(left.from_coeffs(jnp.array([1., 2.])), (-1., 1.))], domain)
    g = Chebfun([_Piece(right.from_coeffs(jnp.array([3., -1., 2.])), (-1., 1.))], domain)
    f = f.set_point_values(jnp.array([10., 20.]))
    g = g.set_point_values(jnp.array([2., 3.]))
    endpoints = jnp.array([-1., 1.])
    assert jnp.array_equal((f+g)(endpoints), jnp.array([12., 23.]))
    assert jnp.array_equal((f-g)(endpoints), jnp.array([8., 17.]))
    assert abs(float((f+g)(0.))-2.) < 1e-14
    assert type((f+g).funs[0].tech) is left
