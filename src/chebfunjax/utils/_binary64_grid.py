"""Scoped gradual-underflow grid for bounded blowup localization.

Provenance
----------
MATLAB source : @fun/detectEdge.m (findBlowup/zoomIn); installed R2025b linspace.m
Chebfun commit: 7574c77
Independently derived integer IEEE arithmetic; fresh MATLAB oracle still required.
"""
import jax
import jax.numpy as jnp

from chebfunjax.utils._binary64 import _divide_binary64_by_positive_integer


def _units(value):
    """Decode bounded binary64 to signed integer multiples of 2**-1074."""
    bits = jax.lax.bitcast_convert_type(value, jnp.uint64)
    exponent = ((bits >> jnp.uint64(52)) & jnp.uint64(2047)).astype(jnp.int64)
    fraction = bits & jnp.uint64((1 << 52)-1)
    mantissa = jnp.where(exponent == 0, fraction, fraction | jnp.uint64(1 << 52))
    magnitude = mantissa << jnp.maximum(exponent-1, 0).astype(jnp.uint64)
    signed = magnitude.astype(jnp.int64)
    return jnp.where((bits >> jnp.uint64(63)) != 0, -signed, signed)


def _rounded_units(value):
    """Round exact signed integer quantum count to binary64, ties to even."""
    magnitude = jnp.abs(value).astype(jnp.uint64)
    length = 64-jax.lax.clz(magnitude).astype(jnp.int64)
    shift = jnp.maximum(length-53, 0).astype(jnp.uint64)
    retained = magnitude >> shift
    discarded = magnitude & ((jnp.uint64(1) << shift)-jnp.uint64(1))
    half = jnp.uint64(1) << (jnp.maximum(shift, jnp.uint64(1))-jnp.uint64(1))
    increment = ((shift != 0) & ((discarded > half)
                 | ((discarded == half) & ((retained & jnp.uint64(1)) != 0))))
    rounded = (retained + increment.astype(jnp.uint64)) << shift
    rounded_length = 64-jax.lax.clz(rounded).astype(jnp.int64)
    exponent = jnp.maximum(rounded_length-52, 1)
    fraction = (rounded >> (exponent-1).astype(jnp.uint64)) & jnp.uint64((1 << 52)-1)
    normal = (exponent.astype(jnp.uint64) << jnp.uint64(52)) | fraction
    bits = jnp.where(rounded_length <= 52, rounded, normal)
    bits = bits | jnp.where(value < 0, jnp.uint64(1 << 63), jnp.uint64(0))
    return jax.lax.bitcast_convert_type(bits, jnp.float64)


def tiny_source_grid(a, b, n):
    """Source linspace for ordered finite tiny endpoints, n in {4,15,50}.

    Caller contract: -2**-1020 <= a < b <= 2**-1020; JAX x64 enabled.
    Each intermediate signed integer is strictly within int64. Supports jit
    with static n. Caller must guard the narrow range; this is not general
    software floating point. Endpoint bits are preserved including -0.

    Provenance
    ----------
    MATLAB source : @fun/detectEdge.m; installed MATLAB R2025b linspace.m
    Chebfun commit: 7574c77
    Independently derived IEEE integer arithmetic for this bounded range.
    """
    if n not in (4, 15, 50):
        raise ValueError("only locator source grid sizes are supported")
    if not jax.config.x64_enabled:
        raise ValueError("binary64 grid requires x64")
    a, b = jnp.asarray(a, dtype=jnp.float64), jnp.asarray(b, dtype=jnp.float64)
    ua, ub = _units(a), _units(b)
    indices = jnp.arange(n, dtype=jnp.int64)
    # MATLAB nonsymmetric branch: separate rounded subtraction, product,
    # quotient, and addition. Integer quantum counts retain subnormals.
    width = _rounded_units(ub-ua)
    product = _rounded_units(indices*_units(width))
    quotient = _divide_binary64_by_positive_integer(product, n-1)
    standard = _rounded_units(ua+_units(quotient))
    # MATLAB uses division before multiplication for symmetric endpoints.
    factor = _divide_binary64_by_positive_integer(b, n-1)
    multipliers = 2*indices-(n-1)
    factor_units = _units(factor)
    symmetric = _rounded_units(multipliers*factor_units)
    # Source multiplication by positive zero retains the multiplier sign.
    bits = jax.lax.bitcast_convert_type(symmetric, jnp.uint64)
    zero_sign = jnp.where((multipliers < 0) & (factor_units == 0),
                          jnp.uint64(1 << 63), jnp.uint64(0))
    symmetric = jax.lax.bitcast_convert_type(bits | zero_sign, jnp.float64)
    result = jnp.where(ua == -ub, symmetric, standard)
    return result.at[0].set(a).at[-1].set(b)


def locator_source_grid(a, b, n):
    """Host dispatch only: preserve existing normal-range locator grids.

    Provenance
    ----------
    MATLAB source : @fun/detectEdge.m; installed MATLAB R2025b linspace.m
    Chebfun commit: 7574c77
    The bounded integer path handles near-zero gradual underflow only.
    """
    if -float.fromhex("0x1p-1020") <= a < b <= float.fromhex("0x1p-1020"):
        return tiny_source_grid(a, b, n)
    return jnp.linspace(a, b, n)
