"""Bitwise source arithmetic and JIT/AD controls for split C1 transforms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1


def _source_vals2coeffs(values):
    n = values.shape[0]
    if n <= 1:
        return jnp.asarray(values, dtype=jnp.complex128 if
                           jnp.iscomplexobj(values) else jnp.float64)
    k = jnp.arange(n, dtype=jnp.float64)
    weights = 2.0 * jnp.exp(1j * k * jnp.pi / (2.0 * n))
    weights = weights.reshape((n,) + (1,) * (values.ndim - 1))
    if jnp.iscomplexobj(values):
        return (_source_vals2coeffs(jnp.real(values))
                + 1j * _source_vals2coeffs(jnp.imag(values)))
    mirrored = jnp.concatenate([values[::-1], values]).astype(jnp.complex128)
    coeffs_complex = jnp.fft.ifft(mirrored, axis=0)[:n] * weights
    coeffs_complex = coeffs_complex.at[0].multiply(0.5)
    coeffs = jnp.real(coeffs_complex)
    flipped = values[::-1]
    is_even = jnp.max(jnp.abs(values - flipped), axis=0) == 0
    is_odd = jnp.max(jnp.abs(values + flipped), axis=0) == 0
    k_rows = jnp.arange(n).reshape((n,) + (1,) * (coeffs.ndim - 1))
    sym = jnp.where((k_rows % 2 == 1) & is_even, 0.0, coeffs)
    sym = jnp.where((k_rows % 2 == 0) & is_odd, 0.0, sym)
    delta = jnp.where(jnp.isfinite(coeffs), sym - coeffs, 0.0)
    return coeffs + jax.lax.stop_gradient(delta)


def _source_coeffs2vals(coeffs):
    n = coeffs.shape[0]
    if jnp.iscomplexobj(coeffs):
        return (_source_coeffs2vals(jnp.real(coeffs))
                + 1j * _source_coeffs2vals(jnp.imag(coeffs)))
    if n <= 1:
        return coeffs.astype(jnp.float64)
    k = jnp.arange(2 * n, dtype=jnp.float64)
    weights = jnp.exp(-1j * k * jnp.pi / (2.0 * n)) / 2.0
    weights = weights.at[0].set(1.0)
    weights = weights.at[n].set(0.0)
    weights = weights.at[n + 1:].multiply(-1.0)
    weights = weights.reshape((2 * n,) + (1,) * (coeffs.ndim - 1))
    mirrored = jnp.concatenate([
        coeffs,
        jnp.ones((1,) + coeffs.shape[1:], dtype=jnp.float64),
        coeffs[-1:0:-1],
    ]).astype(jnp.complex128)
    values_complex = jnp.fft.fft(mirrored * weights, axis=0)
    values = jnp.real(values_complex[n - 1::-1])
    is_even = jnp.max(jnp.abs(coeffs[1::2]), axis=0, initial=0.0) == 0
    is_odd = jnp.max(jnp.abs(coeffs[0::2]), axis=0, initial=0.0) == 0
    flipped = values[::-1]
    sym = jnp.where(is_even, (values + flipped) / 2.0, values)
    sym = jnp.where(is_odd, (values - flipped) / 2.0, sym)
    delta = jnp.where(jnp.isfinite(values), sym - values, 0.0)
    return values + jax.lax.stop_gradient(delta)


def _bits(values):
    return np.asarray(values).view(np.uint64)


@pytest.mark.parametrize(
    "n", [0, 1, 2, 16, 17, 65, 513, 2049, 5792, 16385, 23041, 23169]
)
def test_split_transforms_match_source_arithmetic_bitwise(n):
    rng = np.random.default_rng(2026101509 + n)
    real = rng.standard_normal((n, 4))
    if n:
        real[:, 0] = 1.0
        real[:, 1] = (real[:, 1] + real[::-1, 1]) / 2.0
        real[:, 2] = (real[:, 2] - real[::-1, 2]) / 2.0
    for complex_input in (False, True):
        values = (real + 1j * rng.standard_normal((n, 4))
                  if complex_input else real)
        values = jnp.asarray(values)
        expected_coeffs = _source_vals2coeffs(values)
        actual_coeffs = Chebtech1.vals2coeffs(values)
        np.testing.assert_array_equal(_bits(actual_coeffs),
                                      _bits(expected_coeffs))

        coeffs = jnp.asarray(
            rng.standard_normal((n, 4))
            + (1j * rng.standard_normal((n, 4)) if complex_input else 0.0)
        )
        expected_values = _source_coeffs2vals(coeffs)
        actual_values = Chebtech1.coeffs2vals(coeffs)
        np.testing.assert_array_equal(_bits(actual_values),
                                      _bits(expected_values))


@pytest.mark.parametrize("n", [2, 17])
@pytest.mark.parametrize("complex_input", [False, True])
def test_split_transforms_keep_jit_and_jvp_behavior(n, complex_input):
    values = jnp.linspace(-0.75, 1.25, n)
    coeffs = jnp.cos(jnp.arange(n, dtype=jnp.float64) / 3.0)
    tangent = jnp.linspace(1.0, -0.5, n)
    if complex_input:
        values = values + 0.25j * values[::-1]
        coeffs = coeffs + 0.125j * coeffs[::-1]
        tangent = tangent + 0.375j * tangent[::-1]

    for transform, reference, point in (
        (Chebtech1.vals2coeffs, _source_vals2coeffs, values),
        (Chebtech1.coeffs2vals, _source_coeffs2vals, coeffs),
    ):
        expected = reference(point)
        actual = transform(point)
        np.testing.assert_array_equal(_bits(actual), _bits(expected))
        jitted = jax.jit(transform)(point)
        np.testing.assert_allclose(jitted, expected, rtol=2e-14, atol=2e-14)

        _, expected_tangent = jax.jvp(reference, (point,), (tangent,))
        _, actual_tangent = jax.jvp(transform, (point,), (tangent,))
        np.testing.assert_array_equal(_bits(actual_tangent),
                                      _bits(expected_tangent))
