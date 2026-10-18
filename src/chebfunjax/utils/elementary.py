"""Elementary numerical adapters for real and complex JAX arrays.

Independently stated logarithmic identity, not MATLAB built-in internals.
"""

import jax
import jax.numpy as jnp

from chebfunjax.utils._signed_gradual import gradual_signed_divide
from chebfunjax.utils.numeric_sign import real_sign


def _sin_cos_numeric(value, *, cosine=False):
    """Retain the captured nonfinite real-axis SIN/COS imaginary zero.

    Provenance
    ----------
    MATLAB source: @treeVar/treeVar.m numeric SIN/COS dispatch.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Independent R2025b scalar and mixed-array captures establish positive
    imaginary zero for nonfinite real inputs on the exact real axis. This
    is a builtin boundary convention, not a complex limit at infinity.
    Preserve JAX's real component and all finite/off-axis arithmetic. The
    word-aware axis test excludes nonzero subnormal imaginary components.
    """
    value = jnp.asarray(value)
    result = jnp.cos(value) if cosine else jnp.sin(value)
    if not jnp.iscomplexobj(value):
        return result
    real, imag = jnp.real(value), jnp.imag(value)
    selected = ~jnp.isfinite(real) & (real_sign(imag) == 0)
    return jax.lax.complex(
        jnp.real(result), jnp.where(selected, jnp.zeros_like(imag), jnp.imag(result)))


def _asinh_real(value):
    # |asinh(x)-x| <= |x|^3/6. For |x| < 2^-26 this is below
    # half the binary64 spacing, including the gradual underflow range.
    return jnp.where(jnp.abs(value) < 2.0**-26, value, jnp.arcsinh(value))


@jax.custom_jvp
def _asinh_complex_real_axis(value):
    """Use real ASINH on the exact real axis, retaining complex trace storage.

    Provenance
    ----------
    MATLAB source: @chebfun/asinh.m and @treeVar/treeVar.m builtin dispatch.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    The identity asinh(x+i0)=asinh(x) supplies the real component. Independent
    R2025b scalar/array captures establish positive imaginary zero, including
    infinite and NaN real inputs. The opaque builtin is not reproduced here.
    Normal nonzero imaginary components retain JAX arithmetic. For binary64
    subnormal imaginary components and finite real x, omitted quadratic and
    cubic terms are below representable spacing: real=asinh(x), imag=y/hypot(1,x).
    The division preserves gradual underflow. This does not claim universally
    correctly rounded transcendental values or subnormal derivative accuracy.
    """
    real, imag = jnp.real(value), jnp.imag(value)
    result = jnp.arcsinh(value)
    real_value = _asinh_real(real)
    axis = jax.lax.complex(real_value, jnp.zeros_like(imag))
    if imag.dtype == jnp.float64:
        magnitude = jax.lax.bitcast_convert_type(imag, jnp.uint64) & jnp.uint64(0x7fffffffffffffff)
        tiny = (magnitude > 0) & (magnitude < jnp.uint64(0x0010000000000000)) & jnp.isfinite(real)
        tiny_value = jax.lax.complex(
            real_value, gradual_signed_divide(imag, jnp.hypot(1., real)))
        result = jnp.where(tiny, tiny_value, result)
    return jnp.where(real_sign(imag) == 0, axis, result)


@_asinh_complex_real_axis.defjvp
def _asinh_complex_real_axis_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    ordinary = tangent / jnp.sqrt(1 + value * value)
    # A fixed positive scale cancels algebraically; stop_gradient preserves
    # that identity at every derivative order without differentiating max/abs.
    real, imag = jnp.real(value), jnp.imag(value)
    scale = jax.lax.stop_gradient(jnp.maximum(1., jnp.maximum(jnp.abs(real), jnp.abs(imag))))
    large = jnp.isfinite(scale) & (scale > jnp.sqrt(jnp.finfo(real.dtype).max / 2.))
    if real.dtype == jnp.float64:
        divide = gradual_signed_divide
    else:
        divide = jnp.divide
    scaled_value = jax.lax.complex(divide(real, scale), divide(imag, scale))
    scaled_tangent = jax.lax.complex(
        divide(jnp.real(tangent), scale), divide(jnp.imag(tangent), scale))
    scaled_real, scaled_imag = jnp.real(scaled_value), jnp.imag(scaled_value)
    # Retain the side of the imaginary-axis cut when normalization underflows
    # a component. Complex+real addition would erase a negative imaginary zero.
    square_imag = jnp.copysign(
        jnp.abs(2. * scaled_real * scaled_imag),
        jnp.copysign(1., real) * jnp.copysign(1., imag))
    argument = jax.lax.complex(
        scaled_real**2 - scaled_imag**2 + divide(1., scale)**2, square_imag)
    root = jnp.sqrt(argument)
    # XLA complex sqrt can choose +imag on a negative-real argument with
    # negative imaginary zero; enforce the principal root's signed side.
    root = jax.lax.complex(jnp.real(root), jnp.copysign(jnp.abs(jnp.imag(root)), square_imag))
    scaled = scaled_tangent / root
    return _asinh_complex_real_axis(value), jnp.where(large, scaled, ordinary)


def _asinh_numeric(value):
    """Apply real-axis and gradual-underflow ASINH adaptations.

    Provenance
    ----------
    MATLAB source: @chebfun/asinh.m and @treeVar/treeVar.m numeric dispatch.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """
    value = jnp.asarray(value)
    if jnp.iscomplexobj(value):
        return _asinh_complex_real_axis(value)
    return _asinh_real(value)


def _atanh_log1p_real(x):
    """Compute real-domain atanh by cancellation-resistant log1p identities.

    Real values outside[-1,1] yield NaN so the existing Chebfun elementwise
    wrapper can retry complex arithmetic. Complex inputs keep existing JAX
    arctanh. No approximation/coefficient fit or bound is changed here.

    Provenance
    ----------
    MATLAB source : @chebfun/atanh.m, @chebfun/acoth.m (builtin consumers)
    Chebfun commit: 7574c77
    Independent logarithmic identities; not MATLAB builtin implementation.
    This binary64 numerical adaptation is qualified against high precision
    and original MATLAB Chebfun composition assertions.

    """
    x = jnp.asarray(x)
    if jnp.iscomplexobj(x):
        return jnp.arctanh(x)
    a = jnp.abs(x)
    small = 0.5 * jnp.log1p(2.0 * a + 2.0 * a * a / (1.0 - a))
    large = 0.5 * jnp.log1p(2.0 * a / (1.0 - a))
    magnitude = jnp.where(a < 0.5, small, large)
    result = jnp.copysign(magnitude, x)
    # atanh(x)-x = x^3/3+... is below half a binary64 ulp throughout
    # this general tiny-argument range. Preserve signedzero/subnormal bits.
    return jnp.where(a < 2.0**-28, x, result)
