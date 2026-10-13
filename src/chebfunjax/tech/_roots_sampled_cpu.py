"""Reuse CPU sampled-subdivision grid and recurrence compilation shapes.

Provenance
----------
MATLAB source: @chebtech/roots.m, @chebtech/clenshaw.m, @chebtech2/chebpts.m.
Chebfun commit: 7574c77.
The binary64 division body is the existing independently derived integer
quotient/remainder implementation in utils/_binary64.py. Only host integer
normalization metadata becomes dynamic. Its global API remains unchanged.
The existing fdlibm sine sequence and original-n final FFT are retained.
"""
from functools import partial

import jax
import jax.numpy as jnp

from chebfunjax.tech.chebtech import _vals2coeffs_jax
from chebfunjax.utils._fdlibm_sine import _sin_cheb_angles


def divide_normalized(x, k, d):
    """Exact division body with prevalidated normalized denominator metadata.

    Private caller supplies k=floor(log2(divisor)), d=divisor<<(52-k), with
    divisor=2*(n-1), 513<n<=4000, and JAX x64 enabled. The floating-point
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


@jax.jit
def affine_points(x, a, b):
    # Preserve the original mapped-points helper's dynamic endpoint boundary.
    return b*(x + 1)/2 + a*(1 - x)/2

@partial(jax.jit, static_argnames=("capacity",))
def dynamic_points(n, k, d, capacity):
    i = jnp.arange(capacity, dtype=jnp.int64)
    m = n - 1
    numerator = (2*i - m).astype(jnp.float64)
    numerator = jnp.where(i < n, numerator, 0.)
    angles = divide_normalized(jnp.pi*numerator, k, d)
    x = _sin_cheb_angles(angles)
    left = affine_points(x, -1., -0.004849834917525)
    right = affine_points(x, -0.004849834917525, 1.)
    j = jnp.arange(2*capacity, dtype=jnp.int64)
    points = jnp.where(j < n, left[jnp.minimum(j, capacity-1)],
                      right[jnp.clip(j - n, 0, capacity-1)])
    return jnp.where(j < 2*n, points, 0.)


@jax.jit
def recurrence_states(coeffs, points, n):
    x2 = 2.0 * points
    degree = n - 1

    def body(i, state):
        (b1, b2) = state
        k = degree - 2 * i
        b2 = coeffs[k] + x2 * b1 - b2
        b1 = coeffs[k - 1] + x2 * b2 - b1
        return (b1, b2)
    (b1, b2) = jax.lax.fori_loop(0, degree // 2, body, (jnp.zeros_like(points), jnp.zeros_like(points)))
    return (b1, b2, jnp.all(jnp.isfinite(coeffs)))

@partial(jax.jit, static_argnames=('n',))
def finish_subdivision(coeffs, points, b1, b2, n):
    points = points[:2 * n]
    (b1, b2) = (b1[:2 * n], b2[:2 * n])
    if (n - 1) % 2:
        (b1, b2) = (coeffs[1] + 2.0 * points * b1 - b2, b1)
    values = coeffs[0] + points * b1 - b2
    return (_vals2coeffs_jax(values[:n]), _vals2coeffs_jax(values[n:]))

@jax.jit
def grid_and_states(coeffs, n, k, d):
    points = dynamic_points(n, k, d, coeffs.shape[0])
    b1, b2, finite = recurrence_states(coeffs, points, n)
    return points, b1, b2, finite
