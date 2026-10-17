"""Shared JAX CDR kernels retain polynomial values, rank order and AD."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.separable_approx import SeparableApprox
from chebfunjax.tech.chebtech import Chebtech2


def _fun(c, r, p, domain=(2., 6., -3., 5.)):
    return Chebfun2(SeparableApprox(
        [Chebtech2.from_coeffs(c[:, j]) for j in range(c.shape[1])],
        [Chebtech2.from_coeffs(r[:, j]) for j in range(r.shape[1])],
        p, domain))


def _arrays(complex_values=False):
    c = jnp.array([[1., 2.], [.5, -.25], [.125, .75]])
    r = jnp.array([[.75, -.5], [2., .125], [-.25, .5]])
    p = jnp.array([.75, -.25])
    if complex_values:
        c = c * (1 + .25j)
        p = p * (1 - .5j)
    return c, r, p


def _oracle(c, r, p, x, y):
    tx = (x-2)/4-(6-x)/4
    ty = (y+3)/8-(5-y)/8
    cv = c[0]+ty[..., None]*c[1]+(2*ty[..., None]**2-1)*c[2]
    rv = r[0]+tx[..., None]*r[1]+(2*tx[..., None]**2-1)*r[2]
    return p[0]*cv[..., 0]*rv[..., 0]+p[1]*cv[..., 1]*rv[..., 1]


@pytest.mark.parametrize('complex_values', [False, True])
def test_polynomial_eager_jit_vmap(complex_values):
    c, r, p = _arrays(complex_values)
    f = _fun(c, r, p)
    x = jnp.array([2., 2.5, 4.25, 6.])
    y = jnp.array([-3., 1., 2., 5.])
    expected = _oracle(c, r, p, x, y)
    for actual in (f(x, y), jax.jit(lambda a, b: f(a, b))(x, y),
                   jax.vmap(lambda a, b: f(a, b))(x, y)):
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(f(x+1j*y), expected, rtol=1e-13, atol=1e-13)


def test_coordinate_and_coefficient_jvp_vjp():
    c, r, p = _arrays()
    x, y = jnp.array([2.5, 4.25]), jnp.array([-.5, 2.])
    def actual(c, x, y):
        return _fun(c, r, p)(x, y)
    def expected(c, x, y):
        return _oracle(c, r, p, x, y)
    args = (c, x, y)
    directions = (jnp.arange(6.).reshape(3, 2)/10, jnp.array([.2, -.3]), jnp.array([-.4, .1]))
    av, ad = jax.jvp(actual, args, directions)
    ev, ed = jax.jvp(expected, args, directions)
    np.testing.assert_allclose(av, ev, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(ad, ed, rtol=1e-13, atol=1e-13)
    _, ap = jax.vjp(actual, *args)
    _, ep = jax.vjp(expected, *args)
    for a, e in zip(ap(jnp.array([.5, -.75])), ep(jnp.array([.5, -.75]))):
        np.testing.assert_allclose(a, e, rtol=1e-13, atol=1e-13)


def test_unequal_lengths_retains_traced_fallback():
    c, r, p = _arrays()
    approx = SeparableApprox([Chebtech2.from_coeffs(c[:, 0]), Chebtech2.from_coeffs(c[:2, 1])],
        [Chebtech2.from_coeffs(r[:, 0]), Chebtech2.from_coeffs(r[:, 1])], p, (2., 6., -3., 5.))
    x, y = jnp.array([2.5, 4.25]), jnp.array([-.5, 2.])
    actual = np.asarray(Chebfun2(approx)(x, y))
    expected = np.asarray(approx._call_traced(x, y))
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize('x,y', [(2.5, 1.), (jnp.array([[2.5], [4.25]]), jnp.array([-.5, 1., 2.]))])
def test_scalar_and_matrix_broadcast(x, y):
    c, r, p = _arrays()
    f = _fun(c, r, p)
    x, y = jnp.asarray(x), jnp.asarray(y)
    # Direct point evaluator avoids the public tensor-grid adapter, which
    # deliberately has a different matrix-product evaluation schedule.
    actual = np.asarray(f._eval_xy(x, y))
    expected = np.asarray(f.approx._call_traced(x, y))
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_empty_rank_retains_traced_result():
    approx = SeparableApprox([], [], jnp.empty((0,)), (2., 6., -3., 5.))
    x, y = jnp.array([2.5, 4.25]), jnp.array([-.5, 2.])
    np.testing.assert_array_equal(Chebfun2(approx)._eval_xy(x, y), jnp.zeros(2))


def test_mixed_real_complex_family_keeps_individual_recurrences(monkeypatch):
    c, r, p = _arrays()
    approx = SeparableApprox(
        [Chebtech2.from_coeffs(c[:, 0]),
         Chebtech2.from_coeffs(c[:, 1].astype(jnp.complex128) + .125j)],
        [Chebtech2.from_coeffs(r[:, 0]), Chebtech2.from_coeffs(r[:, 1])],
        p, (2., 6., -3., 5.))
    original = SeparableApprox._call_traced
    calls = []
    def observed(self, x, y):
        calls.append(True)
        return original(self, x, y)
    monkeypatch.setattr(SeparableApprox, '_call_traced', observed)
    x, y = jnp.array([2.5, 4.25]), jnp.array([-.5, 2.])
    actual = np.asarray(approx._call_array_jax(x, y))
    expected = np.asarray(original(approx, x, y))
    assert calls == [True]
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_nonbinary_domain_mapping_keeps_traced_bits():
    c, r, p = _arrays()
    f = _fun(c, r, p, (.1, 2.7, -4.3, .2))
    x, y = jnp.array([.1, .35, 1.9, 2.7]), jnp.array([-4.3, -2.7, -.1, .2])
    expected = np.asarray(f.approx._call_traced(x, y))
    for value in (f._eval_xy(x, y), jax.jit(lambda a, b: f._eval_xy(a, b))(x, y)):
        assert np.array_equal(np.asarray(value).view(np.uint64), expected.view(np.uint64))


def test_complex_coefficient_and_coordinate_jvp():
    c, r, p = _arrays(True)
    x, y = jnp.array([2.5, 4.25]), jnp.array([-.5, 2.])
    def actual(c, x, y):
        return _fun(c, r, p)._eval_xy(x, y)
    def expected(c, x, y):
        return _oracle(c, r, p, x, y)
    directions = (jnp.ones_like(c)*(.2+.3j), jnp.array([.2, -.3]), jnp.array([-.4, .1]))
    av, ad = jax.jvp(actual, (c, x, y), directions)
    ev, ed = jax.jvp(expected, (c, x, y), directions)
    np.testing.assert_allclose(av, ev, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(ad, ed, rtol=1e-13, atol=1e-13)
