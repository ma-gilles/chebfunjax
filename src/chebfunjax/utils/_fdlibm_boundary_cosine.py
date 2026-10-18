"""Bounded binary64 fdlibm cosine for polynomial endpoint boundary rows.

Provenance
----------
https://netlib.org/fdlibm/s_cos.c, e_rem_pio2.c, k_sin.c, k_cos.c.
Chebfun consumer: @chebfun/pdeSolve.m neumannLeft/neumannRight, pin7574c77.
Only finite |x|<=pi is supported; this is not a global cosine replacement.
Reuse source kernels and retain the special n=1 reduction before medium n=2.

Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
Developed at SunSoft, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this software is freely
 granted, provided that this notice is preserved.
"""

import jax
import jax.numpy as jnp

from chebfunjax.utils._fdlibm_projection_sine import _kernel_sin_tail, _medium_reduce_abs
from chebfunjax.utils._fdlibm_sine import _high_abs, _kernel_cos, _sub


@jax.jit
def _cos_boundary_primal(x):
    """Evaluate the source cosine reduction schedule on the bounded interval."""
    if not jax.config.x64_enabled:
        raise ValueError("Requires binary64 mode")
    x = jnp.asarray(x, jnp.float64)
    ax = jnp.abs(x)
    ix = _high_abs(ax)
    valid = jnp.isfinite(ax) & (ax <= jnp.pi)
    p1 = 1.57079632673412561417e00
    p1t = 6.07710050650619224932e-11
    p2 = 6.07710050630396597660e-11
    p2t = 2.02226624879595063154e-21
    z = _sub(ax, p1)
    ordinary_y0 = _sub(z, p1t)
    ordinary_y1 = _sub(_sub(z, ordinary_y0), p1t)
    near_z = _sub(z, p2)
    near_y0 = _sub(near_z, p2t)
    near_y1 = _sub(_sub(near_z, near_y0), p2t)
    near = ix == jnp.uint64(0x3FF921FB)
    y0 = jnp.where(near, near_y0, ordinary_y0)
    y1 = jnp.where(near, near_y1, ordinary_y1)
    special = -_kernel_sin_tail(y0, y1)
    medium_input = jnp.where(valid & (ix >= 0x4002D97C), ax, 3.0)
    n, m0, m1 = _medium_reduce_abs(medium_input)
    medium = jnp.where(n == 1, -_kernel_sin_tail(m0, m1), -_kernel_cos(m0, m1))
    reduced = jnp.where(ix < 0x4002D97C, special, medium)
    result = jnp.where(ix <= 0x3FE921FB, _kernel_cos(ax, jnp.zeros_like(ax)), reduced)
    return jnp.where(valid, result, jnp.nan)


@jax.custom_jvp
def _cos_boundary_angles(x):
    """Source-rounded bounded cosine with the real analytic derivative.

    Constant branches in the finite-precision source kernel retain primal
    words but must not define the derivatives of the cosine function.
    No derivative contract is provided outside the supported real interval.
    """
    return _cos_boundary_primal(x)


@_cos_boundary_angles.defjvp
def _cos_boundary_angles_jvp(primals, tangents):
    (x,), (tangent,) = primals, tangents
    x = jnp.asarray(x, jnp.float64)
    tangent = jnp.asarray(tangent, jnp.float64)
    valid = jnp.isfinite(x) & (jnp.abs(x) <= jnp.pi)
    derivative = jnp.where(valid, -jnp.sin(x), jnp.nan)
    return _cos_boundary_primal(x), derivative * tangent
