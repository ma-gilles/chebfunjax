"""IEEE-real sign primitive for JAX numeric arrays.

Provenance
----------
Original JAX bitwise implementation of MATLAB numeric SIGN semantics used by
@chebfun/sign.m and tests/chebfun/test_feval.m (Chebfun commit7574c77).
Native sign and constructor outputs were independently captured for CPU review.
This is not a copy of MATLAB's unavailable built-in SIGN implementation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def real_sign(value):
    """Return real numeric sign using IEEE words instead of float compares.

    The CPU XLA backend can flush subnormals during floating comparisons.
    This bitwise implementation preserves ``sign(x)`` for every nonzero
    representable real value, with both signed zeros mapped to zero and NaNs
    propagated unchanged. It supports float16, bfloat16, float32, and float64.
    """
    x = jnp.asarray(value)
    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError(f"real_sign expects a real floating array, got {x.dtype}.")

    if x.dtype == jnp.float64:
        word = jax.lax.bitcast_convert_type(x, jnp.uint64)
        lo = word.astype(jnp.uint32)
        hi = (word >> jnp.uint64(32)).astype(jnp.uint32)
        sign_mask, magnitude_mask = jnp.uint32(1 << 31), jnp.uint32((1 << 31) - 1)
        exponent_mask, fraction_mask = jnp.uint32(0x7ff00000), jnp.uint32(0x000fffff)
        one_bits = jnp.uint64(0x3ff0000000000000)
        sign_bit = jnp.uint64(0x8000000000000000)
        magnitude_hi = hi & magnitude_mask
        lo_nonzero = (lo | (jnp.uint32(0) - lo)) >> jnp.uint32(31)
        hi_nonzero = (magnitude_hi | (jnp.uint32(0) - magnitude_hi)) >> jnp.uint32(31)
        nonzero_bit = lo_nonzero | hi_nonzero
        nonzero_mask = jnp.uint64(0) - nonzero_bit.astype(jnp.uint64)
        is_nan = ((magnitude_hi & exponent_mask) == exponent_mask) & (
            ((magnitude_hi & fraction_mask) != 0) | (lo != 0))
        negative = (hi & sign_mask) != 0
    elif x.dtype == jnp.float32:
        word = jax.lax.bitcast_convert_type(x, jnp.uint32)
        sign_mask = jnp.uint32(1 << 31)
        one_bits = jnp.uint32(0x3f800000)
        sign_bit = jnp.uint32(0x80000000)
        lo = word.astype(jnp.uint16)
        hi = (word >> jnp.uint32(16)).astype(jnp.uint16)
        magnitude_hi = hi & jnp.uint16(0x7fff)
        lo_nonzero = (lo | (jnp.uint16(0) - lo)) >> jnp.uint16(15)
        hi_nonzero = (magnitude_hi | (jnp.uint16(0) - magnitude_hi)) >> jnp.uint16(15)
        nonzero_bit = lo_nonzero | hi_nonzero
        mask16 = jnp.uint16(0) - nonzero_bit
        nonzero_mask = mask16.astype(jnp.uint32) | (mask16.astype(jnp.uint32) << jnp.uint32(16))
        fraction_mask, exponent_mask = jnp.uint16(0x007f), jnp.uint16(0x7f80)
        is_nan = ((magnitude_hi & exponent_mask) == exponent_mask) & (
            ((magnitude_hi & fraction_mask) != 0) | (lo != 0))
        negative = (word & sign_mask) != 0
    elif x.dtype == jnp.float16:
        word = jax.lax.bitcast_convert_type(x, jnp.uint16)
        sign_mask, magnitude_mask = jnp.uint16(1 << 15), jnp.uint16((1 << 15) - 1)
        exponent_mask, fraction_mask = jnp.uint16(0x7c00), jnp.uint16(0x03ff)
        one_bits = jnp.uint16(0x3c00)
        sign_bit = jnp.uint16(0x8000)
        magnitude = word & magnitude_mask
        nonzero_bit = (magnitude | (jnp.uint16(0) - magnitude)) >> jnp.uint16(15)
        nonzero_mask = jnp.uint16(0) - nonzero_bit
        is_nan = ((magnitude & exponent_mask) == exponent_mask) & (
            (magnitude & fraction_mask) != 0)
        negative = (word & sign_mask) != 0
    elif x.dtype == jnp.bfloat16:
        word = jax.lax.bitcast_convert_type(x, jnp.uint16)
        sign_mask, magnitude_mask = jnp.uint16(1 << 15), jnp.uint16((1 << 15) - 1)
        exponent_mask, fraction_mask = jnp.uint16(0x7f80), jnp.uint16(0x007f)
        one_bits = jnp.uint16(0x3f80)
        sign_bit = jnp.uint16(0x8000)
        magnitude = word & magnitude_mask
        nonzero_bit = (magnitude | (jnp.uint16(0) - magnitude)) >> jnp.uint16(15)
        nonzero_mask = jnp.uint16(0) - nonzero_bit
        is_nan = ((magnitude & exponent_mask) == exponent_mask) & (
            (magnitude & fraction_mask) != 0)
        negative = (word & sign_mask) != 0
    else:
        raise TypeError(f"real_sign does not support floating dtype {x.dtype}.")

    signed_one_bits = one_bits | jnp.where(negative, sign_bit,
                                           jnp.zeros_like(sign_bit))
    nonzero_sign_bits = signed_one_bits & nonzero_mask
    result_bits = jnp.where(is_nan, word, nonzero_sign_bits)
    return jax.lax.bitcast_convert_type(result_bits, x.dtype)
