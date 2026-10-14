"""CPU grid compilation reuse with unchanged binary64 division and sine.

Chebfun 7574c77, @chebtech2/chebpts.m. The normalized division function is
moved unchanged from tech/_roots_sampled_cpu.py. Host metadata is integer-only.
"""
from functools import partial

import jax
import jax.numpy as jnp

from chebfunjax.utils._fdlibm_sine import _sin_cheb_angles


def divide_normalized(x, k, d):
    """Exact division body with prevalidated normalized denominator metadata.

    Private caller supplies k=floor(log2(divisor)), d=divisor<<(52-k), with
    divisor=2*(n-1), 2<=n<=32768, and JAX x64 enabled. The floating-point
    quotient/rounding body is unchanged from _binary64.py.
    """
    value = jnp.asarray(x, dtype=jnp.float64)
    bits = jax.lax.bitcast_convert_type(value, jnp.uint64)
    u = jnp.uint64
    sign = bits & u(1 << 63)
    raw_exp = ((bits >> u(52)) & u(0x7FF)).astype(jnp.int64)
    fraction = bits & u((1 << 52) - 1)
    finite_nonzero = (raw_exp != 0x7FF) & ((raw_exp != 0) | (fraction != 0))
    # Sanitize unused zero/special lanes so all integer paths are defined.
    mantissa = jnp.where(raw_exp == 0, fraction, fraction | u(1 << 52))
    mantissa = jnp.where(finite_nonzero, mantissa, u(1 << 52))
    shift = jnp.where(raw_exp == 0,
                      jax.lax.clz(mantissa).astype(jnp.int64) - 11, 0)
    mantissa = mantissa << shift.astype(jnp.uint64)
    effective_exp = jnp.where(raw_exp == 0, 1 - shift, raw_exp)

    lower = mantissa < d
    numerator = jnp.where(lower, mantissa << u(1), mantissa)
    exponent = effective_exp - k - lower.astype(jnp.int64)
    remainder = numerator - d
    quotient = jnp.ones_like(mantissa, dtype=jnp.uint64)

    def step(_, state):
        q, r = state
        doubled = r << u(1)
        bit = doubled >= d
        return ((q << u(1)) | bit.astype(jnp.uint64),
                jnp.where(bit, doubled-d, doubled))

    quotient, remainder = jax.lax.fori_loop(0, 52, step, (quotient, remainder))
    # Normal results: round the 53-bit quotient with the exact remainder.
    twice_remainder = remainder << u(1)
    increment = ((twice_remainder > d)
                 | ((twice_remainder == d) & ((quotient & u(1)) != 0)))
    rounded = quotient + increment.astype(jnp.uint64)
    carry = rounded == u(1 << 53)
    normal_q = jnp.where(carry, rounded >> u(1), rounded)
    normal_e = exponent + carry.astype(jnp.int64)
    normal_bits = ((normal_e.astype(jnp.uint64) << u(52))
                   | (normal_q & u((1 << 52)-1)))

    # Subnormal results round ONCE from the unrounded quotient/remainder.
    # Clamping makes both eager select arms defined for every input lane.
    distance = 1 - exponent
    s = jnp.clip(distance, 1, 53).astype(jnp.uint64)
    retained = quotient >> s
    discarded = quotient & ((u(1) << s)-u(1))
    halfway = u(1) << (s-u(1))
    sub_increment = ((discarded > halfway)
                     | ((discarded == halfway)
                        & ((remainder != 0) | ((retained & u(1)) != 0))))
    sub_bits = retained + sub_increment.astype(jnp.uint64)
    # For s>53 the exact value is strictly below half the minimum subnormal.
    sub_bits = jnp.where(distance > 53, u(0), sub_bits)
    magnitude = jnp.where(exponent >= 1, normal_bits, sub_bits)
    result = sign | magnitude
    # Preserve zeros and infinity; quiet every NaN, retaining its payload.
    special = jnp.where((raw_exp == 0x7FF) & (fraction != 0),
                        bits | u(1 << 51), bits)
    result = jnp.where(finite_nonzero, result, special)
    return jax.lax.bitcast_convert_type(result, jnp.float64)


@partial(jax.jit, static_argnames=("capacity",))
def points(n, k, d, capacity):
    i = jnp.arange(capacity, dtype=jnp.int64)
    numerator = (2 * i - (n - 1)).astype(jnp.float64)
    numerator = jnp.where(i < n, numerator, 0.)
    return _sin_cheb_angles(divide_normalized(jnp.pi * numerator, k, d))
