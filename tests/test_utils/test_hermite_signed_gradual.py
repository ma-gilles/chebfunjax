# uses-numpy: host reference assertions and captured MATLAB fixture inspection.
"""Independent sign/rounding and captured source-normalization controls."""
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._signed_gradual import flip_sign_bits, gradual_signed_divide


@pytest.mark.parametrize('disabled', [False, True])
@pytest.mark.parametrize('negative_left', [False, True])
@pytest.mark.parametrize('negative_right', [False, True])
def test_signed_ties_and_zeros(disabled, negative_left, negative_right):
    sign = np.uint64(0x8000000000000000)
    words = np.asarray([0, 1, 3, 5, 2**52-1, 2**52], dtype=np.uint64)
    a = (words | (sign if negative_left else np.uint64(0))).view(np.float64)
    b = -2.0 if negative_right else 2.0
    expected = np.asarray([float(Fraction.from_float(abs(float(x))) / 2) for x in a])
    expected_words = expected.view(np.uint64) | (sign if negative_left ^ negative_right else np.uint64(0))
    with jax.disable_jit(disabled):
        actual = jax.jit(gradual_signed_divide)(jnp.asarray(a), jnp.asarray(b))
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint64), expected_words)


@pytest.mark.parametrize('disabled', [False, True])
def test_matlab_raw_v_normalization(disabled):
    with np.load((Path(__file__).parent / "fixtures/hermite_source_barycentric.npz")) as f:
        raw, divisor, expected = f['raw_v'], f['normalization_divisor'], f['final_v']
    def normalize(half, maximum):
        folded = jnp.concatenate((half[::-1], flip_sign_bits(half)))
        return gradual_signed_divide(folded, maximum)
    with jax.disable_jit(disabled):
        actual = jax.jit(normalize)(jnp.asarray(raw), jnp.asarray(divisor))
    # Exact captured-input scalar division contract, not a full constructor claim.
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint64), expected.view(np.uint64))


def test_normal_signed_derivatives():
    assert float(jax.grad(lambda x: gradual_signed_divide(x, -2.0))(4.0)) == -0.5
    assert float(jax.grad(lambda x: gradual_signed_divide(-4.0, x))(2.0)) == 1.0
