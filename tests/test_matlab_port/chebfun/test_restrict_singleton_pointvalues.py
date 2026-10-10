"""Stored singleton-column pointValues restore under scalar restriction."""
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('complex_values', [False, True])
def test_singleton_point_values_restore(transpose, complex_values):
    f = cj.chebfun(lambda x: x-.3)
    values = jnp.asarray([[2.], [3.]])
    if complex_values:
        values = values + 1j*jnp.asarray([[4.], [5.]])
    f = f.set_point_values(values)
    if transpose:
        f = f.transpose()
    g = f.restrict([-1., 0., 1.])
    assert g.is_transposed == transpose
    np.testing.assert_array_equal(np.asarray(g.point_values).reshape(-1)[[0,2]], values.reshape(-1))
    x = jnp.asarray([-.75, -.25, .25, .75])
    np.testing.assert_allclose(g(x), x-.3, rtol=0, atol=1e-14)

def test_singleton_sign():
    f = cj.chebfun(lambda x: x-.3).set_point_values(jnp.asarray([[-2.], [3.]]))
    g = f.sign()
    np.testing.assert_array_equal(g.point_values, [-1., 0., 1.])
    np.testing.assert_array_equal(g(jnp.asarray([-.5, .5])), [-1., 1.])

def test_retained_random_switching_sign():
    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.domain import Domain
    path = Path(__file__).parent/'fixtures/randomswitching_raw_sign_e70.npz'
    with np.load(path) as z:
        breaks = z['breaks'].copy()
        coeffs = z['coeffs0'].copy()
        pv = z['point_values'].copy()
    assert coeffs.shape == (177,) and pv.shape == (2,1)
    f = Chebfun(funs=[_Piece.from_coeffs(jnp.asarray(coeffs), *breaks)],domain=Domain(breaks))
    f = f.set_point_values(jnp.asarray(pv))
    g = f.sign()
    assert len(g.funs) == 44
    np.testing.assert_array_equal(g.point_values[1:-1], jnp.zeros(43))
    x = jnp.linspace(.013,39.987,201)
    np.testing.assert_array_equal(g(x), jnp.sign(f(x)))
