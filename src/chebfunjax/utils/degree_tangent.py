"""JAX degree tangent with exact quadrant handling for binary64 inputs.

Provenance
----------
MATLAB source : @chebfun3t/tand.m delegates to MATLAB's built-in tand.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Representation adaptation: real angles are reduced in degrees before radian
conversion; exact captured degree values preserve signed zeros, unit values,
and signed pole infinities. Full MATLAB libm bit identity is not claimed.
"""

import jax
import jax.numpy as jnp
from jax import lax

from chebfunjax.utils._gradual import gradual_positive_multiply
from chebfunjax.utils._signed_gradual import flip_sign_bits

_SCALE = jnp.asarray(jnp.pi / 180.0, dtype=jnp.float64)
_SMALL_DEGREES = jnp.asarray(1.0e-7, dtype=jnp.float64)
_SIGN = jnp.uint64(0x8000000000000000)
_MAG = jnp.uint64(0x7FFFFFFFFFFFFFFF)
_POS_INF = jnp.uint64(0x7FF0000000000000)


def _magnitude64(x):
    bits = lax.bitcast_convert_type(x, jnp.uint64)
    return lax.bitcast_convert_type(bits & _MAG, jnp.float64)


def _signed_scale_small(x):
    """Scale one binary64 component without flushing subnormal data."""
    bits = lax.bitcast_convert_type(x, jnp.uint64)
    magnitude = lax.bitcast_convert_type(bits & _MAG, jnp.float64)
    scaled = gradual_positive_multiply(magnitude, _SCALE)
    negative = (bits & _SIGN) != 0
    return jnp.where(negative, flip_sign_bits(scaled), scaled)


def _real_degree_tangent(x):
    bits = lax.bitcast_convert_type(x, jnp.uint64)
    negative_input = (bits & _SIGN) != 0
    magnitude = _magnitude64(x)
    q360_magnitude = jnp.fmod(magnitude, 360.0)
    q180_magnitude = jnp.fmod(magnitude, 180.0)
    q180 = jnp.where(negative_input, -q180_magnitude, q180_magnitude)

    # Reduce in degrees before conversion. Near a pole, keep its exact degree
    # offset to avoid cancellation in x*pi/180. Reducing |x| and restoring the
    # sign avoids the low-bit loss from mod(-epsilon, 180) == 180-epsilon.
    folded = jnp.where(
        q180 > 90.0,
        q180 - 180.0,
        jnp.where(q180 < -90.0, q180 + 180.0, q180),
    )
    ordinary = jnp.tan(folded * _SCALE)
    offset = jnp.where(q180 >= 0.0, q180 - 90.0, q180 + 90.0)
    near_pole = -jnp.cos(offset * _SCALE) / jnp.sin(offset * _SCALE)
    abs_q180 = _magnitude64(q180)
    use_cotangent = (abs_q180 >= 45.0) & (abs_q180 <= 135.0)
    result = jnp.where(use_cotangent, near_pole, ordinary)

    # Captured source values at exact multiples of 45 degrees.
    exact_one = (q180 == 45.0) | (q180 == -135.0)
    exact_minus_one = (q180 == -45.0) | (q180 == 135.0)
    result = jnp.where(exact_one, 1.0, result)
    result = jnp.where(exact_minus_one, -1.0, result)

    exact_pole = abs_q180 == 90.0
    pole_negative = (q360_magnitude == 270.0) ^ negative_input
    pole_bits = _POS_INF | jnp.where(pole_negative, _SIGN, jnp.uint64(0))
    pole_value = lax.bitcast_convert_type(pole_bits, jnp.float64)
    result = jnp.where(exact_pole, pole_value, result)

    exact_zero = q180 == 0.0
    zero_negative = jnp.where(
        q360_magnitude == 180.0, ~negative_input, negative_input
    )
    zero_bits = jnp.where(zero_negative, _SIGN, jnp.uint64(0))
    zero_value = lax.bitcast_convert_type(zero_bits, jnp.float64)
    result = jnp.where(exact_zero, zero_value, result)

    # At |x| <= 1e-7 degrees, tan(pi*x/180) differs from its linear term by
    # relative at most (pi*x/180)^2/3 < 1.1e-18. The gradual product preserves
    # correctly rounded subnormal components that ordinary CPU arithmetic
    # flushes.
    tiny = _magnitude64(x) <= _SMALL_DEGREES
    return jnp.where(tiny, _signed_scale_small(x), result)


def _complex_degree_tangent(x):
    xr, xi = jnp.real(x), jnp.imag(x)
    ibits = lax.bitcast_convert_type(xi, jnp.uint64)
    imag_is_zero = (ibits & _MAG) == 0
    real_result = _real_degree_tangent(xr)
    real_as_complex = lax.complex(real_result, jnp.zeros_like(real_result))

    xrbits = lax.bitcast_convert_type(xr, jnp.uint64)
    negative_real = (xrbits & _SIGN) != 0
    q180_magnitude = jnp.fmod(_magnitude64(xr), 180.0)
    q180 = jnp.where(negative_real, -q180_magnitude, q180_magnitude)
    folded = jnp.where(
        q180 > 90.0,
        q180 - 180.0,
        jnp.where(q180 < -90.0, q180 + 180.0, q180),
    )
    imag_radians = xi * _SCALE
    ordinary_arg = lax.complex(folded * _SCALE, imag_radians)
    ordinary = jnp.tan(ordinary_arg)

    # tan(90 degrees + w) = -cot(w). Reduce the real component around the
    # represented pole before conversion so neighboring floats stay distinct.
    offset = jnp.where(q180 >= 0.0, q180 - 90.0, q180 + 90.0)
    offset_arg = lax.complex(offset * _SCALE, imag_radians)
    near_pole = -1.0 / jnp.tan(offset_arg)
    abs_q180 = _magnitude64(q180)
    use_cotangent = (abs_q180 >= 45.0) & (abs_q180 <= 135.0)
    result = jnp.where(use_cotangent, near_pole, ordinary)

    # MATLAB's captured built-in returns NaN at exact real pole degrees even
    # when the complex argument has a nonzero imaginary component.
    exact_real_pole = abs_q180 == 90.0
    nan_value = lax.complex(jnp.full_like(xr, jnp.nan), jnp.full_like(xi, jnp.nan))
    result = jnp.where(exact_real_pole, nan_value, result)

    small = jnp.maximum(_magnitude64(xr), _magnitude64(xi)) <= _SMALL_DEGREES
    # For z=a+ib in radians, the componentwise leading-term correction is
    # bounded by (a^2 + 3*b^2)/3. At the component limit this is <4.1e-18.
    tiny = lax.complex(_signed_scale_small(xr), _signed_scale_small(xi))
    result = jnp.where(small, tiny, result)
    return jnp.where(imag_is_zero, real_as_complex, result)


@jax.custom_jvp
def degree_tangent(value):
    """Evaluate tangent with an argument in degrees.

    Exact source angles use degree-space reductions and exact equality; no
    proximity tolerance snaps a nearby value to a pole or special angle.
    Qualified inputs are binary64 real or complex values.
    """
    value = jnp.asarray(value)
    if jnp.iscomplexobj(value):
        return _complex_degree_tangent(value.astype(jnp.complex128))
    return _real_degree_tangent(value.astype(jnp.float64))


@degree_tangent.defjvp
def _degree_tangent_jvp(primals, tangents):
    (value,), (value_dot,) = primals, tangents
    result = degree_tangent(value)
    # Exact-value primal branches do not erase the analytic derivative.
    derivative = _SCALE * (1.0 + result * result)
    return result, derivative * value_dot
