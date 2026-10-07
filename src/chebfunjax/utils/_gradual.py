"""Diagnostic JAX binary64 gradual-underflow primitives, positive scalar field.

Provenance
----------
MATLAB source : hermpts.m/hermpts_asy0 and hermpts_asy
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Representation adaptation: each exp/divide/multiply result is separately rounded
and encoded as IEEE754 binary64. This is not an alternative final log-weight.
Source libm exponential bit identity is not claimed.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.custom_batching import custom_vmap

_FRACTION = jnp.uint64(0x000FFFFFFFFFFFFF)
_EXPONENT = jnp.uint64(0x7FF0000000000000)
_TINY = jnp.finfo(jnp.float64).tiny
_LN2_HI = float.fromhex("0x1.62e42fee00000p-1")
_MIN_LOG = -1076 * _LN2_HI  # below half the least subnormal: guaranteed round to zero
_LN2_LO = 1.90821492927058770002e-10


def _positive_parts(value):
    """Decode subnormal bits before floating arithmetic can flush the operand."""
    bits = lax.bitcast_convert_type(value, jnp.uint64)
    fraction = bits & _FRACTION
    subnormal_or_zero = (bits & _EXPONENT) == 0
    # uint fraction <=2**52-1 converts exactly to binary64 normal integers.
    sub_m, sub_e = jnp.frexp(fraction.astype(jnp.float64))
    normal_m, normal_e = jnp.frexp(value)
    return (
        jnp.where(subnormal_or_zero, sub_m, normal_m),
        jnp.where(subnormal_or_zero, sub_e - 1074, normal_e),
    )


def _positive_pack(mantissa, exponent):
    """Round positive m*2**e to binary64, including subnormal output bits."""
    m, e = jnp.frexp(mantissa)
    total = e + exponent
    # For a subnormal result, units of2**-1074 lie in[0,2**52].
    units = jnp.ldexp(m, jnp.clip(total + 1074, 0, 52))
    integer = jnp.rint(units).astype(jnp.uint64)
    integer = jnp.where((total < -1074) | (m == 0), jnp.uint64(0), integer)
    encoded = lax.bitcast_convert_type(integer, jnp.float64)
    normal = jnp.ldexp(m, total)
    return jnp.where(total < -1021, encoded, normal)


def _integer_parts(value):
    """Exact positive significand/exponent, normalized to53 integer bits."""
    bits = lax.bitcast_convert_type(value, jnp.uint64)
    field = (bits >> 52) & jnp.uint64(2047)
    fraction = bits & _FRACTION
    sig = jnp.where(field == 0, fraction, fraction | jnp.uint64(1 << 52))
    exponent = jnp.where(field == 0, -1074, field.astype(jnp.int32) - 1075)
    shift = jnp.maximum(lax.clz(sig).astype(jnp.int32) - 11, 0)
    return sig << shift.astype(jnp.uint64), exponent - shift


def _rounded_quotient_units(numerator, denominator):
    """Exact RNE quotient in units2^-1074 where result is <=2*tiny."""
    n, ne = _integer_parts(numerator)
    d, de = _integer_parts(denominator)
    k = ne - de + 1074
    first = (n >= d).astype(jnp.uint64)
    remainder = n - jnp.where(first != 0, d, jnp.uint64(0))

    def step(i, state):
        q, r = state
        twice = r << jnp.uint64(1)
        digit = (twice >= d).astype(jnp.uint64)
        next_r = twice - jnp.where(digit != 0, d, jnp.uint64(0))
        next_q = (q << jnp.uint64(1)) | digit
        return jnp.where(i < k, next_q, q), jnp.where(i < k, next_r, r)

    q, r = lax.fori_loop(0, 52, step, (first, remainder))
    twice = r << jnp.uint64(1)
    increment = (twice > d) | ((twice == d) & ((q & jnp.uint64(1)) != 0))
    rounded = q + increment.astype(jnp.uint64)
    # Normalized n/d is in[.5,2); k=-1 needs only comparison with1.
    rounded = jnp.where(k == -1, (n > d).astype(jnp.uint64), rounded)
    rounded = jnp.where((k < -1) | (n == 0), jnp.uint64(0), rounded)
    return lax.bitcast_convert_type(rounded, jnp.float64), k <= 52


def _wide_product(a, b):
    """Exact106-bit product as high/low64-bit limbs; no float product."""
    mask = jnp.uint64(0xFFFFFFFF)
    a0, a1 = a & mask, a >> jnp.uint64(32)
    b0, b1 = b & mask, b >> jnp.uint64(32)
    p00, p01, p10, p11 = a0 * b0, a0 * b1, a1 * b0, a1 * b1
    middle = (p00 >> jnp.uint64(32)) + (p01 & mask) + (p10 & mask)
    lo = (p00 & mask) | ((middle & mask) << jnp.uint64(32))
    hi = p11 + (p01 >> jnp.uint64(32)) + (p10 >> jnp.uint64(32)) + (middle >> jnp.uint64(32))
    return hi, lo


def _rounded_product_units(left, right):
    """Exact RNE product in units2^-1074 where result is <=2*tiny."""
    a, ae = _integer_parts(left)
    b, be = _integer_parts(right)
    hi, lo = _wide_product(a, b)
    shift = -(ae + be + 1074)
    low_shift = jnp.clip(shift, 1, 64).astype(jnp.uint64)
    low_mask = (jnp.uint64(1) << low_shift) - jnp.uint64(1)
    low_rem = lo & low_mask
    low_half = jnp.uint64(1) << (low_shift - jnp.uint64(1))
    low_q = (lo >> low_shift) | (hi << (jnp.uint64(64) - low_shift))
    low_q = jnp.where(shift == 64, hi, low_q)
    low_above, low_tie = low_rem > low_half, low_rem == low_half
    high_shift = jnp.clip(shift - 64, 1, 63).astype(jnp.uint64)
    high_rem = hi & ((jnp.uint64(1) << high_shift) - jnp.uint64(1))
    high_half = jnp.uint64(1) << (high_shift - jnp.uint64(1))
    high_q = hi >> high_shift
    high_above = (high_rem > high_half) | ((high_rem == high_half) & (lo != 0))
    high_tie = (high_rem == high_half) & (lo == 0)
    q = jnp.where(shift <= 64, low_q, high_q)
    above = jnp.where(shift <= 64, low_above, high_above)
    tie = jnp.where(shift <= 64, low_tie, high_tie)
    rounded = q + (above | (tie & ((q & jnp.uint64(1)) != 0))).astype(jnp.uint64)
    rounded = jnp.where((shift > 106) | (a == 0) | (b == 0), jnp.uint64(0), rounded)
    return lax.bitcast_convert_type(rounded, jnp.float64), shift >= 53


@jax.custom_jvp
def gradual_exp_negative(value):
    """Preserve normal exp arithmetic; explicitly encode tiny finite outputs.

    Provenance
    ----------
    MATLAB source : hermpts.m, ASY weight and barycentric normalization
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    JAX binary64 representation adaptation; source libm bit identity is not claimed.
    """
    value = jnp.asarray(value, dtype=jnp.float64)
    ordinary = jnp.exp(value)
    # Bound integers for nonfinite/extreme arguments before conversion.
    safe = jnp.clip(value, _MIN_LOG, 0.0)
    k = jnp.floor(safe / (_LN2_HI + _LN2_LO)).astype(jnp.int32)
    # Truncated high part makes k*high exact for the bounded integer k.
    # Low part restores log(2); barriers preserve separate rounded operations.
    high = lax.optimization_barrier(k.astype(jnp.float64) * _LN2_HI)
    reduced = lax.optimization_barrier(safe - high)
    reduced = reduced - k.astype(jnp.float64) * _LN2_LO
    mantissa = jnp.exp(reduced)
    tiny_result = _positive_pack(mantissa, k)
    tiny_result = jnp.where(value < _MIN_LOG, 0.0, tiny_result)
    use = jnp.isfinite(value) & (value <= 0) & (ordinary < _TINY)
    return jnp.where(use, tiny_result, ordinary)


@gradual_exp_negative.defjvp
def _exp_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    y = gradual_exp_negative(x)
    return y, y * dx


@custom_vmap
def _gradual_positive_divide_primal(numerator, denominator):
    """Separately round division; decode a tiny positive numerator first."""
    numerator, denominator = jnp.broadcast_arrays(numerator, denominator)
    ordinary = numerator / denominator
    a, ae = _positive_parts(numerator)
    b, be = _positive_parts(denominator)
    # Preserve the broadcast denominator as a vector division operand.
    # Without this boundary XLA hoists 1/b and introduces an extra rounding.
    # Scope: normalized mantissas only; all source selection/packing is unchanged.
    b = lax.optimization_barrier(jnp.broadcast_to(b, a.shape))
    encoded = _positive_pack(a / b, ae - be)
    exact, eligible = _rounded_quotient_units(numerator, denominator)
    encoded = jnp.where(eligible, exact, encoded)
    bits = lax.bitcast_convert_type(numerator, jnp.uint64)
    sub = ((bits & _EXPONENT) == 0) & ((bits & _FRACTION) != 0)
    use = (
        ((ordinary < _TINY) | sub)
        & ((bits >> 63) == 0)
        & (denominator >= _TINY)
        & jnp.isfinite(denominator)
    )
    return jnp.where(use, encoded, ordinary)


@_gradual_positive_divide_primal.def_vmap
def _divide_batch(axis_size, in_batched, numerator, denominator):
    """Expose every mapped denominator entry before the division boundary.

    Provenance
    ----------
    MATLAB source : hermpts.m, hermpts_asy0 barycentric division
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    JAX batching adaptation preserves elementwise division semantics.
    """
    args = (numerator, denominator)
    logical = [x.shape[1:] if batched else x.shape for x, batched in zip(args, in_batched)]
    shape = jnp.broadcast_shapes(*logical)
    full_shape = (axis_size,) + shape
    arrays = []
    for x, batched, item_shape in zip(args, in_batched, logical):
        reshape = (
            (axis_size if batched else 1,) + (1,) * (len(shape) - len(item_shape)) + item_shape
        )
        arrays.append(jnp.broadcast_to(jnp.reshape(x, reshape), full_shape))
    a, b = arrays
    # A scalar barrier remains unmapped under the default batching rule.
    # Here b explicitly includes the batch dimension, including nested maps.
    b = lax.optimization_barrier(b)
    return _gradual_positive_divide_primal(a, b), True


@jax.custom_jvp
def gradual_positive_divide(numerator, denominator):
    """Source division with explicit batching of its rounding boundary.

    Provenance
    ----------
    MATLAB source : hermpts.m, hermpts_asy0 barycentric division
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    JAX adaptation: broadcasting the denominator across mapped axes prevents
    a separately rounded scalar reciprocal replacing elementwise division.
    This is not a general all-backend correctly-rounded arithmetic claim.
    """
    return _gradual_positive_divide_primal(numerator, denominator)



@gradual_positive_divide.defjvp
def _divide_jvp(primals, tangents):
    a, b = primals
    da, db = tangents
    y = gradual_positive_divide(a, b)
    return y, (da - y * db) / b


@jax.custom_jvp
def gradual_positive_multiply(left, right):
    """Round multiplication after decoding a possibly subnormal left operand.

    Provenance
    ----------
    MATLAB source : hermpts.m, ASY weight and barycentric normalization
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    JAX binary64 representation adaptation; source libm bit identity is not claimed.
    """
    left, right = jnp.broadcast_arrays(left, right)
    ordinary = left * right
    a, ae = _positive_parts(left)
    b, be = _positive_parts(right)
    encoded = _positive_pack(a * b, ae + be)
    exact, eligible = _rounded_product_units(left, right)
    encoded = jnp.where(eligible, exact, encoded)
    bits = lax.bitcast_convert_type(left, jnp.uint64)
    sub = ((bits & _EXPONENT) == 0) & ((bits & _FRACTION) != 0)
    use = ((ordinary < _TINY) | sub) & ((bits >> 63) == 0) & (right >= _TINY) & jnp.isfinite(right)
    return jnp.where(use, encoded, ordinary)


@gradual_positive_multiply.defjvp
def _multiply_jvp(primals, tangents):
    a, b = primals
    da, db = tangents
    return gradual_positive_multiply(a, b), da * b + a * db


def sequential_weight_stages(x, ders, normalizer):
    """Diagnostic source ordering, no cross-stage logarithmic simplification.

    Provenance
    ----------
    MATLAB source : hermpts.m, ASY weight and barycentric normalization
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    JAX binary64 representation adaptation; source libm bit identity is not claimed.
    """
    square = x * x
    numerator = gradual_exp_negative(-square)
    denominator = ders * ders
    raw = gradual_positive_divide(numerator, denominator)
    final = gradual_positive_multiply(raw, normalizer)
    return numerator, raw, final
