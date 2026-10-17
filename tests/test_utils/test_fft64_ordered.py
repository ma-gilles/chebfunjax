"""Independent Fourier-matrix, differentiation and dispatch controls."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._fft64 import inverse32_pairs
from chebfunjax.utils.transforms import vals2coeffs


def dct_matrix():
    k = np.arange(33)
    weights = np.ones(33)
    weights[[0, -1]] = 0.5
    return np.cos(np.pi * k[:, None] * (32 - k)[None, :] / 32) * (
        weights[:, None] * weights[None, :] / 16
    )


@pytest.mark.parametrize("shape", [(33,), (33, 1)])
def test_jit_real_column_against_defining_dct(shape):
    x = (((np.arange(33) * 97 + 31) % 257 - 128) / 129).reshape(shape)
    actual = jax.jit(vals2coeffs)(jnp.asarray(x))
    np.testing.assert_allclose(
        actual, (dct_matrix() @ x.reshape(33)).reshape(shape), rtol=0, atol=8e-15
    )


def test_jvp_and_vjp_are_the_linear_transform():
    x = jnp.linspace(-0.3, 0.7, 33)
    tangent = jnp.sin(jnp.arange(33, dtype=jnp.float64))
    cotangent = jnp.cos(jnp.arange(33, dtype=jnp.float64))
    _, forward = jax.jit(lambda a, b: jax.jvp(vals2coeffs, (a,), (b,)))(x, tangent)
    reverse = jax.jit(lambda a, b: jax.vjp(vals2coeffs, a)[1](b)[0])(x, cotangent)
    np.testing.assert_allclose(forward, dct_matrix() @ np.asarray(tangent), rtol=0, atol=8e-15)
    np.testing.assert_allclose(reverse, dct_matrix().T @ np.asarray(cotangent), rtol=0, atol=8e-15)


def test_inverse32_general_complex_fourier_definition():
    z = np.arange(32) / 31 + 1j * np.cos(np.arange(32))
    pairs = jnp.asarray(np.stack([z.real, z.imag], axis=1))
    actual = np.asarray(jax.jit(inverse32_pairs)(pairs))
    expected = np.fft.ifft(z) * 32
    np.testing.assert_allclose(actual[:, 0] + 1j * actual[:, 1], expected, rtol=0, atol=8e-15)


@pytest.mark.parametrize("kind", ["float32", "complex", "batch4", "length17"])
def test_other_transform_shapes_keep_fourier_semantics(kind):
    n = 17 if kind == "length17" else 33
    x = jnp.arange(n, dtype=jnp.float64) / n
    if kind == "float32":
        x = x.astype(jnp.float32)
    elif kind == "complex":
        x = x + 1j * jnp.sin(x)
    elif kind == "batch4":
        x = jnp.stack([x, x * x, x**3, jnp.sin(x)], axis=1)
    y = vals2coeffs(x)
    native_fft = jnp.fft.ifft(jnp.concatenate([x[:0:-1], x[:-1]]), axis=0)[:n]
    native_fft = native_fft.at[1 : n - 1].multiply(2)
    expected = native_fft if jnp.iscomplexobj(x) else jnp.real(native_fft)
    np.testing.assert_allclose(y, expected, rtol=0, atol=2e-7 if kind == "float32" else 2e-15)
