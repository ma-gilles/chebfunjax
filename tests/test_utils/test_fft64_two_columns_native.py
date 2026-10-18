"""Two-column native C2R plan: Chebfun7574c77, MATLAB R2025b CPU.

Independent source input/output fixtures qualify general33x2dispatch. They
are expected results only; production numerics never access these files.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.transforms import vals2coeffs


def _cases():
    with np.load(Path(__file__).parent / "fixtures/fft64_two_columns_native.npz") as data:
        return data["values"], data["coeffs"]


@pytest.mark.parametrize("case", range(10))
def test_jit_two_column_coefficients_match_independent_native_words(case):
    values, expected = _cases()
    actual = np.asarray(jax.jit(vals2coeffs)(jnp.asarray(values[case])))
    assert actual.shape == (33, 2)
    np.testing.assert_array_equal(actual.view(np.uint64), expected[case].view(np.uint64))


def test_two_column_jvp_vjp_match_defining_dct():
    # Same independent DCT definition/bound as existing scalar FFT64 controls.
    k = np.arange(33)
    weights = np.ones(33)
    weights[[0, -1]] = 0.5
    matrix = np.cos(np.pi * k[:, None] * (32 - k)[None, :] / 32) * (
        weights[:, None] * weights[None, :] / 16
    )
    values, _ = _cases()
    x, direction = jnp.asarray(values[0]), jnp.asarray(values[1])
    cotangent = jnp.cos(jnp.arange(66, dtype=jnp.float64)).reshape(33, 2)
    _, forward = jax.jit(lambda a, b: jax.jvp(vals2coeffs, (a,), (b,)))(x, direction)
    reverse = jax.jit(lambda a, b: jax.vjp(vals2coeffs, a)[1](b)[0])(x, cotangent)
    np.testing.assert_allclose(forward, matrix @ np.asarray(direction), rtol=0, atol=8e-15)
    np.testing.assert_allclose(reverse, matrix.T @ np.asarray(cotangent), rtol=0, atol=8e-15)
