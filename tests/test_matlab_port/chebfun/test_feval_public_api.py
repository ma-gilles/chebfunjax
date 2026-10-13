"""Public evaluation entrypoint from @chebfun/feval.m, pin7574c77.

These supplement the numbered native test predicates with documented function
handle and composition dispatch and literal endpoint/side syntax.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain


def test_function_handle_arguments_and_jit():
    x = jnp.asarray([-.5, 0., .5])
    def operation(q, scale):
        return scale * q**2
    expected = np.asarray([.75, 0., .75])
    np.testing.assert_array_equal(cj.feval(operation, x, 3.), expected)
    np.testing.assert_array_equal(jax.jit(lambda q: cj.feval(operation, q, 3.))(x), expected)


def test_matrix_and_complex_query_values():
    f = Chebfun.from_coeffs(jnp.asarray([1., 2.]))
    x = jnp.asarray([[0., .5], [-.5, .25]]) + .125j
    expected = 1 + 2*np.asarray(x)
    np.testing.assert_allclose(cj.feval(f, x), expected, rtol=0., atol=4*np.finfo(float).eps)
    np.testing.assert_allclose(jax.jit(lambda q: cj.feval(f, q))(x), expected,
                               rtol=0., atol=4*np.finfo(float).eps)


def test_endpoint_and_one_sided_documented_syntax():
    f = Chebfun(funs=[_Piece.from_coeffs(jnp.asarray([-1.]), -1., 0.),
                     _Piece.from_coeffs(jnp.asarray([1.]), 0., 1.)],
                domain=Domain((-1., 0., 1.)))
    for alias in ('left', 'start', '-'):
        assert float(cj.feval(f, alias)) == -1.
    for alias in ('right', 'end', '+'):
        assert float(cj.feval(f, alias)) == 1.
    for side, expected in [('left', -1.), ('-', -1.), ('right', 1.), ('+', 1.)]:
        assert float(cj.feval(f, 0., side)) == expected
    with pytest.raises(ValueError) as caught:
        cj.feval(f, 0., 'start')
    assert caught.value.identifier == 'CHEBFUN:CHEBFUN:feval:leftRightChar'


def test_chebfun_composition_dispatch():
    f = Chebfun.from_coeffs(jnp.asarray([.5, 0., .5]))  # x**2
    g = Chebfun.from_coeffs(jnp.asarray([0., .5]))  # x/2
    composed = cj.feval(f, g)
    x = jnp.asarray([-.8, -.2, .3, .9])
    np.testing.assert_allclose(composed(x), np.asarray(x)**2/4,
                               rtol=0., atol=10*np.finfo(float).eps)
