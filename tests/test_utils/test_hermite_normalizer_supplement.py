# uses-numpy: host reference assertions and captured MATLAB fixture inspection.
"""Supplemental source-normalizer coverage; expected-only captured fixture.

Provenance
----------
MATLAB source : hermpts.m, hermpts_asy odd/even folding and normalization
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._signed_gradual import flip_sign_bits, source_barycentric_normalize


@pytest.mark.parametrize("disabled", [False, True])
def test_actual_source_normalizer(disabled):
    fixture = Path(__file__).parent / "fixtures/hermite_source_barycentric.npz"
    with np.load(fixture) as data:
        raw, expected = data["raw_v"], data["final_v"]

    def action(half):
        folded = jnp.concatenate((half[::-1], flip_sign_bits(half)))
        return source_barycentric_normalize(folded)

    with jax.disable_jit(disabled):
        actual = jax.jit(action)(jnp.asarray(raw))
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("disabled", [False, True])
def test_normalization_preserves_signed_zeros_and_tiny_values(disabled):
    # Maximum is exactly one: source division must preserve every input bit.
    words = np.asarray([0, 1, 2**52 - 1, 0x3ff0000000000000,
                        0x8000000000000000, 0x8000000000000001,
                        0xbff0000000000000], dtype=np.uint64)
    with jax.disable_jit(disabled):
        actual = jax.jit(source_barycentric_normalize)(jnp.asarray(words.view(np.float64)))
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint64), words)
