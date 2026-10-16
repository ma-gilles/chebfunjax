"""Source AD extraction controls, independent of a conditioned BVP solve.

Provenance
----------
MATLAB source: @chebop/linearize.m, @linBlock/toCoeff.m.
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import _ad_piecewise_coefficients

DOMAIN = (0., 4e-7, 1.)


def extract(op, orders, *, require_linear=True, domain=DOMAIN):
    zeros = [chebfun(0., domain=domain) for _ in orders]
    return _ad_piecewise_coefficients(op, zeros, orders, require_linear=require_linear)


def test_small_highest_coefficient_is_exact_on_both_panels():
    c, affine = extract(lambda us: [-1e-8*us[0].diff(2)-us[0].diff()], [2])
    for x in (jnp.array([0., 1e-7, 4e-7]), jnp.array([4e-7, .5, 1.])):
        for f, expected in zip(c[0][0], (0., -1., -1e-8)):
            npt.assert_array_equal(f(x), jnp.full_like(x, expected))
        npt.assert_array_equal(affine[0](x), jnp.zeros_like(x))


def test_coupled_orders_and_affine_terms():
    x = chebfun('x', domain=DOMAIN)
    c, affine = extract(lambda us: [1e-8*us[0].diff(2)+2*us[1].diff()+3*us[0]+5,
                                    7*us[0].diff()-us[1]+2*x], [2, 1])
    points = jnp.array([.1, .4, .9])
    for row, expected in zip(c, (((3., 0., 1e-8), (0., 2.)),
                                ((0., 7., 0.), (-1., 0.)))):
        for block, values in zip(row, expected):
            assert len(block) == len(values)
            for f, v in zip(block, values):
                npt.assert_array_equal(f(points), jnp.full_like(points, v))
    npt.assert_allclose(affine[0](points), 5., rtol=0, atol=0)
    npt.assert_allclose(affine[1](points), 2*points, rtol=1e-15, atol=1e-15)


def test_product_derivative_uses_leibniz_rule():
    domain = (2., 3., 5.)
    x = chebfun('x', domain=domain)
    c, _ = extract(lambda us: [(us[0]*x).diff(2)], [2], domain=domain)
    points = jnp.array([2., 2.5, 3.])
    for f, expected in zip(c[0][0], (0., 2., points)):
        npt.assert_allclose(f(points), expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('u0,v0', [(2., 3.), (4., -2.)])
def test_supplied_newton_callback_uses_current_primal(u0, v0):
    u = chebfun(u0, domain=DOMAIN)
    v = chebfun(v0, domain=DOMAIN)
    h = 1e-6

    def callback(ws):
        a, b = u+h*ws[0], v+h*ws[1]
        return [(a.diff(2)+a*b-u*v)/h,
                (b.diff()+a*a-u*u)/h]

    c, affine = extract(callback, [2, 1], require_linear=False)
    points = jnp.array([.1, .5, .9])
    for row, expected in zip(c, (((v0, 0., 1.), (u0, 0.)),
                                ((2*u0, 0., 0.), (0., 1.)))):
        for block, values in zip(row, expected):
            for f, value in zip(block, values):
                npt.assert_allclose(f(points), value, rtol=2e-15, atol=2e-15)
    for f in affine:
        npt.assert_allclose(f(points), 0., rtol=0, atol=0)


def test_nonlocal_block_is_rejected():
    with pytest.raises(TypeError, match='coefficient realization'):
        extract(lambda us: [us[0].cumsum()], [0])


def test_constant_equation_and_scalar_jax_output():
    c, affine = extract(lambda us: [jnp.asarray(3.)], [2])
    points = jnp.array([0., .5, 1.])
    assert len(c[0][0]) == 3
    for f in c[0][0]:
        npt.assert_array_equal(f(points), jnp.zeros_like(points))
    npt.assert_array_equal(affine[0](points), jnp.full_like(points, 3.))


def test_coefficient_extra_breakpoints_require_reconciliation():
    multiplier = chebfun('x', domain=(0., .25, 1.))
    with pytest.raises(ValueError, match='domain reconciliation'):
        extract(lambda us: [us[0]*multiplier], [0], domain=(0., 1.))


def test_discontinuous_multiplier_uses_panel_endpoint_limits():
    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.domain import Domain
    domain = (0., .5, 1.)
    multiplier = Chebfun(funs=[_Piece.from_coeffs(jnp.array([2.]), 0., .5),
                              _Piece.from_coeffs(jnp.array([7.]), .5, 1.)],
                         domain=Domain(domain))
    c, _ = extract(lambda us: [us[0]*multiplier], [0], domain=domain)
    npt.assert_array_equal(c[0][0][0](jnp.array([0., .25, .5])), np.full(3, 2.))
    npt.assert_array_equal(c[0][0][0](jnp.array([.5, .75, 1.])), np.full(3, 7.))
