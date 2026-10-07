# uses-numpy: host reference assertions and captured MATLAB fixture inspection.
"""Exact Fraction controls for V2's 106-bit product limb boundary.

Representation adaptation of hermpts.m sequential multiplication, pinned
Chebfun 7574c77680d7e82b79626300bf255498271a72df. No captured outputs.
Standalone: add the sealed V2 directory to PYTHONPATH before collection.
"""
import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._gradual import gradual_positive_multiply


@pytest.mark.parametrize('disabled', [False, True])
@pytest.mark.parametrize('shift', [63, 64, 65])
@pytest.mark.parametrize('odd_tie', [False, True])
def test_product_guard_sticky_at_limb_boundary(disabled, shift, odd_tie):
    if odd_tie:
        a_sig = (1 << 52) + (1 << (shift - 52))
        b_sig = (1 << 52) + (1 << 51)
    else:
        a_sig = (1 << 52) + (1 << 31)
        b_sig = (1 << 52) + (1 << (shift - 32))
    quotient, remainder = divmod(a_sig * b_sig, 1 << shift)
    assert remainder == 1 << (shift - 1)
    assert bool(quotient & 1) == odd_tie
    # +/- one significand unit supplies below/above-half sticky cases.
    left = np.asarray([math.ldexp(a_sig + d, -1074) for d in (-1, 0, 1)])
    right = math.ldexp(b_sig, -shift)
    expected = np.asarray([
        float(Fraction.from_float(float(a)) * Fraction.from_float(right))
        for a in left
    ])
    with jax.disable_jit(disabled):
        actual = jax.jit(gradual_positive_multiply)(jnp.asarray(left), jnp.asarray(right))
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint64), expected.view(np.uint64))
