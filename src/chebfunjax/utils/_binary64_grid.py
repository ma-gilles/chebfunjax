"""Scoped gradual-underflow grid for bounded blowup localization.

Provenance
----------
MATLAB source : @fun/detectEdge.m (findBlowup/zoomIn); installed R2025b linspace.m
Chebfun commit: 7574c77
The normal branch follows licensed R2025b operation order; retained native
edge traces qualify that consumer. The unchanged tiny branch uses independently
derived integer IEEE arithmetic with retained MATLAB oracle controls. The
adjacent subnormal transition retains its historical backend fallback and
remains unqualified for native bitwise parity.
"""
import jax
import jax.numpy as jnp

from chebfunjax.utils._binary64 import _divide_binary64_by_positive_integer

# Reuse the qualified IEEE quotient primitive; cache by array shape/divisor.
_locator_quotient = jax.jit(_divide_binary64_by_positive_integer, static_argnums=1)


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


def _normal_source_grid(a, b, n):
    """Finite normal-range locator grid in licensed source operation order.

    Provenance
    ----------
    MATLAB source: R2025b toolbox/matlab/elmat/linspace.m.
    Chebfun consumer: @fun/detectEdge.m; Chebfun commit: 7574c77.
    The caller supplies ordered finite binary64 endpoints and n >= 2.
    The separate integer path retains gradual underflow near zero.
    """
    barrier = jax.lax.optimization_barrier
    a, b = jnp.asarray(a, jnp.float64), jnp.asarray(b, jnp.float64)
    n1 = n - 1
    k = jnp.arange(n, dtype=jnp.float64)
    width = barrier(b-a)
    product = barrier(k*width)
    standard = a + _locator_quotient(product, n1)
    divided_width = _locator_quotient(width, n1)
    overflow_product = a + barrier(k*divided_width)
    bscaled, ascaled = _locator_quotient(b, n1), _locator_quotient(a, n1)
    overflow_width = barrier(a + barrier(bscaled*k)) - barrier(ascaled*k)
    c = barrier(width*(n1-1))
    ordinary = jnp.where(jnp.isinf(c),
                         jnp.where(jnp.isinf(width), overflow_width,
                                   overflow_product), standard)
    symmetric = barrier(2*k-n1) * bscaled
    y = jnp.where((a == -b) & (n > 2), symmetric, ordinary)
    if n % 2:
        y = y.at[n//2].set(jnp.where(a == -b, 0., y[n//2]))
    y = jnp.where(a == b, a, y)
    return y.at[0].set(a).at[-1].set(b)


def locator_source_grid(a, b, n):
    """Source locator linspace with a separate gradual-underflow path.

    Provenance
    ----------
    MATLAB source : @fun/detectEdge.m; licensed R2025b linspace.m
    Chebfun commit: 7574c77
    Normal grids retain source arithmetic, symmetric and overflow branches.
    This host-only dispatcher is not a fully native linspace implementation.
    The bounded integer path handles near-zero gradual underflow only.
    """
    if -float.fromhex("0x1p-1020") <= a < b <= float.fromhex("0x1p-1020"):
        return tiny_source_grid(a, b, n)
    # Outside the proven integer-kernel range, retain the existing grid
    # for the transition where a 50-point locator may require subnormal
    # intermediate arithmetic. Source parity is unqualified in this band;
    # JAX CPU arithmetic may flush such intermediates to zero.
    if max(abs(a), abs(b)) < 49 * float.fromhex("0x1p-1022"):
        return jnp.linspace(a, b, n)
    return _normal_source_grid(a, b, n)
