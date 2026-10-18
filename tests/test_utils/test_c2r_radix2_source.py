"""Independent real DCT-I family controls; native source vals2coeffs.m."""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils._ifft_normalization import source_ifft
from chebfunjax.utils.transforms import coeffs2vals, vals2coeffs

EPS = jnp.finfo(jnp.float64).eps


def dct_matrix(n):
    k = jnp.arange(n, dtype=jnp.float64)
    w = jnp.ones(n).at[0].set(.5).at[-1].set(.5)
    return (jnp.cos(jnp.pi * k[:, None] * (n - 1 - k)[None, :] / (n - 1))
            * (2 / (n - 1)) * w[:, None] * w[None, :])


def check(actual, expected, tol=100 * EPS):
    assert actual.shape == expected.shape
    assert bool(jnp.max(jnp.abs(actual - expected), initial=0.) < tol)


@pytest.mark.parametrize("n", [17, 65, 257, 4097, 8193, 16385])
@pytest.mark.parametrize("layout", ["vector", "column", "batch"])
def test_polynomial_and_independent_spectrum(n, layout):
    x = jnp.cos(jnp.arange(n - 1, -1, -1) * jnp.pi / (n - 1))
    # Analytic degree-seven polynomial, using T3/T7 in monomial form.
    t3 = 4 * x**3 - 3 * x
    t7 = 64 * x**7 - 112 * x**5 + 56 * x**3 - 7 * x
    v = .5 + .25 * x - .125 * t3 + .0625 * t7
    c = jnp.zeros(n).at[0].set(.5).at[1].set(.25).at[3].set(-.125).at[7].set(.0625)
    if layout == "column":
        v, c = v[:, None], c[:, None]
    elif layout == "batch":
        v, c = jnp.stack([v, v[::-1]], axis=1), jnp.stack([c, c * (-1.)**jnp.arange(n)], axis=1)
    check(jax.jit(vals2coeffs)(v), c)
    check(coeffs2vals(vals2coeffs(v)), v)


@pytest.mark.parametrize("n", [3, 17, 65, 257, 16385])
def test_dc_nyquist_and_exact_parity(n):
    for values, degree in [(jnp.ones(n), 0), ((-1.)**jnp.arange(n), n - 1)]:
        expected = jnp.zeros(n).at[degree].set(1.)
        check(vals2coeffs(values), expected)
    v = ((17 * jnp.arange(n) + 3) % 257 - 128) / 128.
    even = v + v[::-1]
    odd = v - v[::-1]
    assert bool(jnp.all(vals2coeffs(even)[1::2] == 0))
    assert bool(jnp.all(vals2coeffs(odd)[::2] == 0))
    check(coeffs2vals(vals2coeffs(v)), v)


@pytest.mark.parametrize("n", [5, 17, 65])
def test_defining_matrix_linear_jvp_vjp_vmap(n):
    v = ((17 * jnp.arange(n) + 3) % 257 - 128) / 128.
    tangent = jnp.sin(jnp.arange(n, dtype=jnp.float64))
    cotangent = jnp.cos(jnp.arange(n, dtype=jnp.float64))
    matrix = dct_matrix(n)
    check(vals2coeffs(v), matrix @ v)
    _, forward = jax.jit(lambda a, b: jax.jvp(vals2coeffs, (a,), (b,)))(v, tangent)
    reverse = jax.jit(lambda a, b: jax.vjp(vals2coeffs, a)[1](b)[0])(v, cotangent)
    check(forward, matrix @ tangent)
    check(reverse, matrix.T @ cotangent)
    batch = jnp.stack([v, tangent])
    check(jax.jit(jax.vmap(vals2coeffs))(batch), batch @ matrix.T)


@pytest.mark.parametrize("kind", ["real_storage", "imaginary_storage", "general"])
def test_complex_storage_and_transverse_tangent_unchanged(kind):
    n = 17
    v = jnp.linspace(.125, .875, n).astype(jnp.complex128)
    if kind == "imaginary_storage":
        v = 1j * v
    elif kind == "general":
        v = v + 1j * jnp.sin(jnp.arange(n))
    tangent = jnp.cos(jnp.arange(n)) + 1j * jnp.sin(jnp.arange(n))
    value, derivative = jax.jvp(vals2coeffs, (v,), (tangent,))
    matrix = dct_matrix(n)
    check(value, matrix @ v)
    check(derivative, matrix @ tangent)


@pytest.mark.parametrize("kind", ["singleton", "float32", "nonpower2", "nan", "inf", "negative_zero"])
def test_fallback_and_zero_contract(kind):
    if kind == "singleton":
        assert float(vals2coeffs(jnp.asarray([2.5]))[0]) == 2.5
    elif kind == "float32":
        v = jnp.linspace(-.75, .75, 17, dtype=jnp.float32)
        c = vals2coeffs(v)
        assert c.dtype == v.dtype
        check(coeffs2vals(c), v, 100 * jnp.finfo(jnp.float32).eps)
    elif kind == "nonpower2":
        v = jnp.linspace(-.75, .75, 18)
        check(vals2coeffs(v), dct_matrix(18) @ v)
    elif kind in ("nan", "inf"):
        v = jnp.ones(17).at[5].set(jnp.nan if kind == "nan" else jnp.inf)
        # Nonfinite FFT outputs need not all be NaN (zero-weight butterflies
        # may retain a finite zero). Preserve the existing generic FFT route.
        tmp = jnp.concatenate([v[:0:-1], v[:-1]])
        expected = jnp.real(source_ifft(tmp))[:17].at[1:-1].multiply(2.)
        actual = vals2coeffs(v)
        assert bool(jnp.all(jnp.isnan(actual) == jnp.isnan(expected)))
        a = jnp.where(jnp.isnan(actual), 0., actual)
        b = jnp.where(jnp.isnan(expected), 0., expected)
        assert bool(jnp.all(a == b))
        assert bool(jnp.all(jnp.signbit(a) == jnp.signbit(b)))
    else:
        c = vals2coeffs(-jnp.zeros(17))
        assert bool(jnp.all(c == 0))
