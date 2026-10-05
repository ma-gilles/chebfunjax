"""Bounded evaluation-map source order and CPU tracing controls.

Provenance
----------
MATLAB source : @mapping/mapping.m (linear), @bndfun/feval.m,
                tests/chebfun/test_power.m (operation31)
Chebfun commit: 7574c77
Independent oracle: exact T1 coefficients; analytic derivative 2/(b-a).
No fresh MATLAB execution; NumPy implements the literal source-map oracle.
"""
import jax
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech2

EPS = np.finfo(float).eps


def _linear_piece(a, b):
    return _Piece(tech=Chebtech2.from_coeffs(jnp.asarray([0., 1.])),
                  interval=(a, b))


def _source_inverse(x, a, b):
    return (x-a)/(b-a)-(b-x)/(b-a)


def test_eager_domain_and_chebfun_retain_source_float_order():
    x = np.asarray([-1.8988567412339663, -1.8581976272355345, -.1, 4.2])
    expected = _source_inverse(x, -2., 7.)
    dom = Domain((-2., 7.))
    f = Chebfun(funs=[_linear_piece(-2., 7.)], domain=dom)
    # Exact backend equality is meaningful for the dyadic width eight.
    # Width nine is covered below at the predeclared map-control bound;
    # CPU XLA lowers /9 to multiplication by a rounded reciprocal.
    dyadic = Domain((-2., 6.))
    np.testing.assert_array_equal(np.asarray(dyadic.inverse_map(jnp.asarray(x))),
                                  _source_inverse(x, -2., 6.))
    np.testing.assert_array_equal(np.asarray(f(jnp.asarray(x))), expected)


def test_traced_piece_and_chebfun_match_source_map():
    x = jnp.asarray([-1.8988567412339663, -1.8581976272355345, -.1, 4.2])
    piece = _linear_piece(-2., 7.)
    f = Chebfun(funs=[piece], domain=Domain((-2., 7.)))
    expected = _source_inverse(np.asarray(x), -2., 7.)
    for actual in (Domain((-2., 7.)).inverse_map(x), piece(x),
                   jax.jit(lambda points: f(points))(x)):
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=0., atol=20*EPS)
    derivative = jax.vmap(jax.grad(lambda point: f(point)))(x)
    np.testing.assert_allclose(np.asarray(derivative), 2./9., rtol=0., atol=20*EPS)


def test_complex_arguments_preserve_inverse_map_and_traced_evaluation():
    x = jnp.asarray([-1.8+.125j, 1.+.75j, 6.8-.25j])
    f = Chebfun(funs=[_linear_piece(-2., 7.)], domain=Domain((-2., 7.)))
    expected = _source_inverse(np.asarray(x), -2., 7.)
    for actual in (f(x), jax.jit(lambda points: f(points))(x)):
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=0., atol=20*EPS)


def test_multipiece_eager_and_traced_inverse_map():
    x = jnp.asarray([-1.9, -.5, .1, .6, 4., 6.9])
    f = Chebfun(funs=[_linear_piece(-2., .5), _linear_piece(.5, 7.)],
                domain=Domain((-2., .5, 7.)))
    expected = np.where(np.asarray(x)<.5,
                        _source_inverse(np.asarray(x), -2., .5),
                        _source_inverse(np.asarray(x), .5, 7.))
    for actual in (f(x), jax.jit(lambda points: f(points))(x)):
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=0., atol=20*EPS)


def test_source31_singular_power_original_bound_under_tracing():
    f = cj.chebfun(lambda x:(x+2.)**-1.2, domain=(-2., 7.),
                   exps=(-1.2, 0.), splitting=True)
    result = f**.77
    x = jnp.asarray(8.8*np.random.RandomState(6178).rand(100)-1.9)
    exact = (x+2.)**(-1.2*.77)
    bound = 10*EPS*float(jnp.max(jnp.abs(exact)))
    error = float(jnp.max(jnp.abs(jax.jit(lambda points:result(points))(x)-exact)))
    assert error < bound
