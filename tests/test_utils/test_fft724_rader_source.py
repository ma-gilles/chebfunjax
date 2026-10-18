"""Native724 source-family controls; no fitted chopping or native word claim."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._fft724 import _inverse181, real_even_inverse724
from chebfunjax.utils._ifft_normalization import source_ifft
from chebfunjax.utils.transforms import _vals2coeffs_jax, coeffs2vals

TOL = 100 * jnp.finfo(jnp.float64).eps
FIXTURE = Path(__file__).parent / "fixtures" / "fft724_native.npz"
CASES = [
    "retained1",
    "retained2",
    "retained3",
    "retained4",
    "retained9",
    "polynomial1",
    "polynomial4",
    "dyadic1",
    "dyadic4",
]


@pytest.mark.parametrize("case", CASES)
def test_native_standalone_source_bound(case):
    with np.load(FIXTURE) as f:
        values = jnp.asarray(f[case + "_values"])
        expected = jnp.asarray(f[case + "_coeffs"])
    actual = _vals2coeffs_jax(values)
    assert actual.shape == expected.shape
    assert actual.dtype == jnp.float64
    assert jnp.max(jnp.abs(actual - expected)) < TOL


@pytest.mark.parametrize("degree", [0, 1, 2, 37, 181, 362])
def test_independent_spectrum(degree):
    c = jnp.zeros(363, jnp.float64).at[degree].set(1.0)
    values = coeffs2vals(c)
    actual = _vals2coeffs_jax(values)
    assert jnp.max(jnp.abs(actual - c)) < TOL


@pytest.mark.parametrize("width", [1, 3, 256, 257])
def test_column_layout_and_chunk_boundary(width):
    k = jnp.arange(363, dtype=jnp.float64)
    v = (jnp.remainder(17 * k + 3, 257) - 128) / 128
    columns = v[:, None] * (1 + jnp.arange(width) % 5)[None, :]
    actual = _vals2coeffs_jax(columns)
    one = _vals2coeffs_jax(v)
    assert jnp.max(jnp.abs(actual - one[:, None] * (1 + jnp.arange(width) % 5))) < TOL


def test_rader_prime_independent_direct_dft():
    # Independent O(n**2) JAX reference, with integer modular phase reduction.
    # No prime FFT, Rader permutation, convolution or phase_pair is reused.
    k = jnp.arange(181)
    x = ((17 * k + 3) % 257 - 128) / 128.0 + 1j * ((31 * k + 7) % 251 - 125) / 128.0
    phase = (2 * jnp.pi / 181) * ((k[:, None] * k[None, :]) % 181)
    expected = jnp.sum(jnp.exp(1j * phase) * x[None, :], axis=1) / 181
    actual = _inverse181(x) / 181
    assert jnp.max(jnp.abs(actual - expected)) < TOL


@pytest.mark.parametrize("symmetric", [False, True])
def test_linear_first_and_second_derivatives(symmetric):
    k = jnp.arange(363, dtype=jnp.float64)
    values = jnp.cos(k / 11.0)
    if symmetric:
        values = values + values[::-1]
    tangent = jnp.sin(k / 7.0)
    primal, derivative = jax.jvp(_vals2coeffs_jax, (values,), (tangent,))
    assert jnp.max(jnp.abs(derivative - _vals2coeffs_jax(tangent))) < TOL
    second = jax.jvp(
        lambda v: jax.jvp(_vals2coeffs_jax, (v,), (tangent,))[1], (values,), (tangent,)
    )[1]
    assert jnp.all(second == 0)
    assert jnp.all(jnp.isfinite(primal))


def _legacy_real(values):
    n = values.shape[0]
    tmp = jnp.concatenate((values[n - 1 : 0 : -1], values[: n - 1]))
    c = jnp.real(source_ifft(tmp))[:n].at[1 : n - 1].multiply(2.0)
    k = jnp.arange(n).reshape((n,) + (1,) * (c.ndim - 1))
    even = jnp.max(jnp.abs(values - values[::-1]), axis=0) == 0
    odd = jnp.max(jnp.abs(values + values[::-1]), axis=0) == 0
    sym = jnp.where((k % 2 == 1) & even, 0.0, c)
    sym = jnp.where((k % 2 == 0) & odd, 0.0, sym)
    return c + jax.lax.stop_gradient(jnp.where(jnp.isfinite(c), sym - c, 0.0))


@pytest.mark.parametrize("special", [jnp.inf, -jnp.inf, jnp.nan])
def test_nonfinite_preserves_legacy_route(special):
    values = jnp.arange(363, dtype=jnp.float64).at[73].set(special)
    actual = _vals2coeffs_jax(values)
    expected = jax.jit(_legacy_real)(values)
    assert jnp.array_equal(actual, expected, equal_nan=True)


def test_float32_and_complex_fallbacks():
    values = jnp.arange(363, dtype=jnp.float32) / 128
    assert jnp.array_equal(_vals2coeffs_jax(values), jax.jit(_legacy_real)(values))
    z = values.astype(jnp.float64) + 1j * (0.25 + jnp.sin(values))
    tmp = jnp.concatenate((z[362:0:-1], z[:362]))
    expected = source_ifft(tmp)[:363].at[1:-1].multiply(2.0)
    assert jnp.array_equal(_vals2coeffs_jax(z), expected)


def test_signed_zero_and_empty_columns():
    for sign in (1.0, -1.0):
        v = jnp.full(363, jnp.copysign(0.0, sign), dtype=jnp.float64)
        assert jnp.all(_vals2coeffs_jax(v) == 0)
    assert real_even_inverse724(jnp.empty((724, 0), jnp.float64)).shape == (724, 0)
