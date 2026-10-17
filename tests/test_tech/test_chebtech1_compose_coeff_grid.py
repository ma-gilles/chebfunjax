"""Analytic controls for native coefficient-grid composition sampling.

Provenance
----------
MATLAB source: @chebtech1/compose.m composeResample1/composeResample2.
Chebfun commit: 7574c77.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1


def _data(array, offset):
    a = np.array([2., 3.]) if array else np.array(2.)
    b = np.array([.25, -.5]) if array else np.array(.25)
    return np.stack((a+offset, b)), a+offset, b


def _forbid_point_sampling(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Callable composition sampled an operand by pointwise evaluation')
    monkeypatch.setattr(Chebtech1, '__call__', forbidden)


@pytest.mark.parametrize('array', [False, True])
def test_unary_analytic_square_without_point_sampling(monkeypatch, array):
    coeffs, a, b = _data(array, 0.)
    f = Chebtech1.from_coeffs(jnp.asarray(coeffs))
    _forbid_point_sampling(monkeypatch)
    result = f.compose(lambda u: u*u)
    expected = np.stack((a*a+b*b/2, 2*a*b, b*b/2))
    assert result.coeffs.shape == expected.shape
    assert np.max(np.abs(np.asarray(result.coeffs)-expected)) < 20*np.finfo(float).eps*np.max(np.abs(expected))
    np.testing.assert_array_equal(np.asarray(f.coeffs), coeffs)


@pytest.mark.parametrize('f_array,g_array', [(False, False), (True, False), (False, True), (True, True)])
def test_binary_analytic_columns_without_point_sampling(monkeypatch, f_array, g_array):
    fc, a, b = _data(f_array, 0.)
    gc, c, d = _data(g_array, 1.)
    f = Chebtech1.from_coeffs(jnp.asarray(fc))
    g = Chebtech1.from_coeffs(jnp.asarray(gc))
    _forbid_point_sampling(monkeypatch)
    result = f.compose(lambda u, v: u+2*v, g)
    expected = np.stack((a+2*c, b+2*d))
    assert result.coeffs.shape == expected.shape
    assert np.max(np.abs(np.asarray(result.coeffs)-expected)) < 20*np.finfo(float).eps*np.max(np.abs(expected))
    np.testing.assert_array_equal(np.asarray(f.coeffs), fc)
    np.testing.assert_array_equal(np.asarray(g.coeffs), gc)
