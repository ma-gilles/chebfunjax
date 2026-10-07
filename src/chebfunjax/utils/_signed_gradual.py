"""Signed binary64 adaptation of the separately rounded Hermite divisions."""
import jax
import jax.numpy as jnp
from jax import lax

from chebfunjax.utils._gradual import gradual_exp_negative, gradual_positive_divide

_SIGN = jnp.uint64(0x8000000000000000)
_MAG = jnp.uint64(0x7fffffffffffffff)
_INF = jnp.uint64(0x7ff0000000000000)
_MIN_NORMAL = jnp.uint64(0x0010000000000000)


@jax.custom_jvp
def gradual_signed_divide(numerator, denominator):
    """Divide finite binary64 numerators by finite normal real denominators.

    Sign and zero bits are assembled without multiplying a tiny result by -1.
    Other operand classes retain native arithmetic. Subnormal derivatives are
    not qualified by this primal representation adaptation.

    Provenance
    ----------
    MATLAB source : hermpts.m, hermpts_asy0 division and hermpts_asy normalization
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    a, b = jnp.broadcast_arrays(jnp.asarray(numerator, dtype=jnp.float64),
                                jnp.asarray(denominator, dtype=jnp.float64))
    ab = lax.bitcast_convert_type(a, jnp.uint64)
    bb = lax.bitcast_convert_type(b, jnp.uint64)
    am, bm = ab & _MAG, bb & _MAG
    magnitude = gradual_positive_divide(lax.bitcast_convert_type(am, jnp.float64),
                                        lax.bitcast_convert_type(bm, jnp.float64))
    result_bits = lax.bitcast_convert_type(magnitude, jnp.uint64) | ((ab ^ bb) & _SIGN)
    supported = (am < _INF) & (bm >= _MIN_NORMAL) & (bm < _INF)
    return jnp.where(supported, lax.bitcast_convert_type(result_bits, jnp.float64), a / b)


@gradual_signed_divide.defjvp
def _signed_divide_jvp(primals, tangents):
    a, b = primals
    da, db = tangents
    y = gradual_signed_divide(a, b)
    return y, (da - y * db) / b


def source_barycentric_half(x, ders):
    """Preserve source exponential then signed division as separate operations.

    Provenance
    ----------
    MATLAB source : hermpts.m, hermpts_asy0
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    return gradual_signed_divide(gradual_exp_negative(-(x**2) / 2.0), ders)


def source_barycentric_normalize(v):
    """Divide by the largest absolute entry, without reciprocal multiplication.

    Provenance
    ----------
    MATLAB source : hermpts.m, hermpts_asy
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    return gradual_signed_divide(v, jnp.max(jnp.abs(v)))


def flip_sign_bits(v):
    """Exact unary sign change including zeros and subnormals (reflection arm).

    Provenance
    ----------
    MATLAB source : hermpts.m, ASY weight and barycentric normalization
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    JAX binary64 representation adaptation; source libm bit identity is not claimed.
    """
    return lax.bitcast_convert_type(lax.bitcast_convert_type(v, jnp.uint64) ^ _SIGN,
                                    jnp.float64)
