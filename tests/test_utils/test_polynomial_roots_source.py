"""Literal R2017a roots preprocessing and independent polynomial controls."""
import jax.numpy as jnp
import pytest

from chebfunjax.utils._polynomial_roots_source import polynomial_roots


def observe_companion(monkeypatch, expected, result):
    calls = []

    def eigvals(actual):
        assert actual.dtype == expected.dtype
        assert jnp.array_equal(actual, expected)
        calls.append(actual)
        return jnp.asarray(result, dtype=jnp.promote_types(actual.dtype, jnp.complex64))

    monkeypatch.setattr(jnp.linalg, 'eigvals', eigvals)
    return calls


def test_real_quadratic_action():
    p = jnp.asarray([1., -3., 2.])
    r = polynomial_roots(p)
    # Distinct well-separated exact roots1,2; 64eps accounts for the tiny
    # companion eigenproblem, not a native MATLAB rounding guarantee.
    eps = jnp.finfo(p.dtype).eps
    assert float(jnp.max(jnp.abs(jnp.sort(jnp.real(r))-jnp.asarray([1., 2.])))) < 64*eps
    assert float(jnp.max(jnp.abs((r-1)*(r-2)))) < 64*eps


def test_complex_quadratic_action():
    p = jnp.asarray([1.+0j, -3j, -2.+0j])
    r = polynomial_roots(p)
    eps = jnp.finfo(jnp.float64).eps
    assert float(jnp.max(jnp.abs((r-1j)*(r-2j)))) < 64*eps
    assert float(jnp.max(jnp.abs(jnp.sort(jnp.imag(r))-jnp.asarray([1., 2.])))) < 64*eps


@pytest.mark.parametrize('p,trailing', [
    ([0., 0., 1., -3., 2.], 0),
    ([1., -3., 2., 0., 0.], 2),
    ([0., 1., -3., 2., 0., 0.], 2),
])
def test_exact_zero_trimming_and_prefix(monkeypatch, p, trailing):
    expected = jnp.asarray([[3., -2.], [1., 0.]])
    calls = observe_companion(monkeypatch, expected, [2., 1.])
    actual = polynomial_roots(jnp.asarray(p))
    assert len(calls) == 1
    assert jnp.array_equal(actual, jnp.asarray([0.]*trailing+[2., 1.]))


def test_allzero():
    assert polynomial_roots(jnp.zeros((4,))).shape == (0,)


def test_nonzero_constant():
    assert polynomial_roots(jnp.asarray(3.)).shape == (0,)


def test_relative_leading_overflow(monkeypatch):
    tiny = jnp.asarray(2.**-1022)
    assert bool(tiny != 0)
    calls = observe_companion(monkeypatch, jnp.asarray([[-1.]]), [-1.])
    assert jnp.array_equal(polynomial_roots(jnp.asarray([tiny, 4., 4.])), jnp.asarray([-1.]))
    assert len(calls) == 1


def test_overflow_with_trailing_zeros(monkeypatch):
    def forbidden(_):
        raise AssertionError('Source constant after overflow removal does not call eig')
    monkeypatch.setattr(jnp.linalg, 'eigvals', forbidden)
    tiny = jnp.asarray(2.**-1022)
    assert bool(tiny != 0)
    assert jnp.array_equal(polynomial_roots(jnp.asarray([tiny, 4., 0., 0.])), jnp.zeros((2,)))


@pytest.mark.parametrize('value', [jnp.nan, jnp.inf, -jnp.inf])
def test_nonfinite_rejected(value):
    with pytest.raises(ValueError, match='finite'):
        polynomial_roots(jnp.asarray([1., value, 1.]))


@pytest.mark.parametrize('shape', [(3,), (1, 3), (3, 1)])
def test_vector_shapes(monkeypatch, shape):
    calls = observe_companion(monkeypatch, jnp.asarray([[3., -2.], [1., 0.]]), [2., 1.])
    r = polynomial_roots(jnp.asarray([1., -3., 2.]).reshape(shape))
    assert len(calls) == 1 and r.shape == (2,)


def test_matrix_rejected():
    with pytest.raises(ValueError, match='vector'):
        polynomial_roots(jnp.ones((2, 2)))


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128])
def test_companion_input_dtype_preserved(monkeypatch, dtype):
    expected = jnp.asarray([[3., -2.], [1., 0.]], dtype=dtype)
    calls = observe_companion(monkeypatch, expected, [2., 1.])
    r = polynomial_roots(jnp.asarray([1., -3., 2., 0.], dtype=dtype))
    assert len(calls) == 1
    assert r.dtype == jnp.promote_types(jnp.dtype(dtype), jnp.complex64)
    assert jnp.array_equal(r, jnp.asarray([0., 2., 1.]))


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128])
@pytest.mark.parametrize('coefficients', [[0., 0., 0.], [3.]])
def test_native_zero_constant_real_class_dtype(dtype, coefficients):
    p = jnp.asarray(coefficients, dtype=dtype)
    r = polynomial_roots(p)
    assert r.shape == (0,)
    assert r.dtype == jnp.real(p).dtype
