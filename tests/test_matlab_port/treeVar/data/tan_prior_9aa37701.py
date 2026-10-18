"""Frozen pre-correction TAN implementation for same-runtime preservation tests.

Provenance
----------
Chebfunjax source commit: 9aa37701e330aeb947e33bbe09fbdbf60ed6d4e1.
Original elementary.py SHA256:
9b6952b70be9618818005cc09df10250c2a00f8334b1b0316b85980302709399.
The three function bodies and decorators below are verbatim from that source.
This is a Python regression reference, not a MATLAB numerical oracle. NaN
signs outside the native correction mask depend on the runtime CPU/compiler;
a snapshot from one CPU does not establish their values on another CPU.
"""
from functools import partial

import jax
import jax.numpy as jnp

from chebfunjax.utils.numeric_sign import real_sign


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _tan_tanh_complex_axes(value, hyperbolic):
    primitive = jnp.tanh if hyperbolic else jnp.tan
    result = primitive(value)
    real, imag = jnp.real(value), jnp.imag(value)
    zero_real, zero_imag = real_sign(real) == 0, real_sign(imag) == 0
    result_real, result_imag = jnp.real(result), jnp.imag(result)
    if hyperbolic:
        # Exclude the origin: its real zero retains the input sign.
        imaginary_axis = zero_real & ~zero_imag & jnp.isfinite(imag)
        result_real = jnp.where(imaginary_axis, jnp.zeros_like(real), result_real)
    else:
        result_real = jnp.where(zero_real & zero_imag, real, result_real)
        result_imag = jnp.where(zero_imag, jnp.zeros_like(imag), result_imag)
    return jax.lax.complex(result_real, result_imag)



@_tan_tanh_complex_axes.defjvp
def _tan_tanh_complex_axes_jvp(hyperbolic, primals, tangents):
    (value,), (tangent,) = primals, tangents
    primitive = jnp.tanh if hyperbolic else jnp.tan
    # Retain the original primitive's analytic derivative and its arithmetic.
    _, derivative = jax.jvp(primitive, (value,), (tangent,))
    return _tan_tanh_complex_axes(value, hyperbolic), derivative



def _tan_numeric(value):
    """Retain captured TAN real-axis component zero conventions.

    Provenance
    ----------
    MATLAB source: @treeVar/treeVar.m numeric TAN dispatch.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Independent R2025b direct-component scalar/mixed-array captures establish
    positive imaginary zero on the exact real axis, including nonfinite real
    inputs, and retain the input real-zero sign when both components are zero.
    Word-aware tests exclude nonzero subnormals. Other primitive words and
    the ordinary analytic derivatives are retained; infinite-imaginary phase
    conventions and minimum-subnormal component losses remain unqualified.
    """
    value = jnp.asarray(value)
    if not jnp.iscomplexobj(value):
        return jnp.tan(value)
    return _tan_tanh_complex_axes(value, False)

