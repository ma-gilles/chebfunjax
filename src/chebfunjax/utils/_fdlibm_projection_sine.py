"""Bounded fdlibm sine extension through finite binary64 |x|<=pi.

Provenance: https://netlib.org/fdlibm/e_rem_pio2.c (medium reduction),
https://netlib.org/fdlibm/k_sin.c (iy=1), https://netlib.org/fdlibm/s_sin.c.
Chebfun consumer: barymat.m at commit 7574c77. No general range reduction.

Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
Developed at SunSoft, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this software is freely
 granted, provided that this notice is preserved.

Uses existing short ordered-arithmetic scans instead of a generic register
interpreter. Numerical values use float arithmetic; no custom JVP. Exact
value and gradient controls are recorded in the linked qualification evidence.
"""

import jax
import jax.numpy as jnp

from chebfunjax.utils._fdlibm_sine import (
    _high_abs,
    _kernel_cos,
    _mul,
    _ordered_arithmetic,
    _sub,
)
from chebfunjax.utils._fdlibm_sine import (
    _sin_cheb_angles as _old_sine,
)


def _kernel_sin_tail(x, y):
    """k_sin.c iy=1; source polynomial and correction in short scans."""
    s1 = -1.66666666666666324348e-01
    s2 = 8.33333333332248946124e-03
    s3 = -1.98412698298579493134e-04
    s4 = 2.75573137070700676789e-06
    s5 = -2.50507602534068634195e-08
    s6 = 1.58969099521155010221e-10
    z = _mul(x, x)
    v = _mul(z, x)
    r = _ordered_arithmetic(
        jnp.full_like(x, s6),
        [
            (0, z),
            (1, s5),
            (0, z),
            (1, s4),
            (0, z),
            (1, s3),
            (0, z),
            (1, s2),
        ],
    )
    # Source: x-((z*(half*y-v*r)-y)-v*S1).
    correction = _ordered_arithmetic(
        y,
        [
            (0, 0.5),
            (2, _mul(v, r)),
            (0, z),
            (2, y),
            (2, _mul(v, s1)),
        ],
    )
    value = _sub(x, correction)
    return jnp.where(_high_abs(x) < 0x3E400000, x, value)


def _medium_reduce_abs(t):
    """Source medium reduction with n in{1,2}, all cancellation levels."""
    n = _ordered_arithmetic(
        t,
        [
            (0, 6.36619772367581382433e-01),
            (1, 0.5),
        ],
    ).astype(jnp.int64)

    # Within this bounded branch fn is1 or2. Each fn*constant is exact
    # binary scaling, so select its literal binary64 value without a kernel.
    def scaled(c):
        return jnp.where(n == 1, jnp.asarray(c, jnp.float64), jnp.asarray(2 * c, jnp.float64))

    p1 = scaled(1.57079632673412561417e00)
    w1 = scaled(6.07710050650619224932e-11)
    p2 = scaled(6.07710050630396597660e-11)
    p2t = scaled(2.02226624879595063154e-21)
    p3 = scaled(2.02226624871116645580e-21)
    p3t = scaled(8.47842766036889956997e-32)

    r1 = _sub(t, p1)
    y01 = _ordered_arithmetic(t, [(2, p1), (2, w1)])
    # Source second correction: t=r1; r2=t-p2;
    # w2=p2t-((t-r2)-p2); y02=r2-w2.
    r2 = _ordered_arithmetic(t, [(2, p1), (2, p2)])
    lost2 = _ordered_arithmetic(r1, [(2, r2), (2, p2)])
    w2 = _sub(p2t, lost2)
    y02 = _sub(r2, w2)
    # Source third correction has the same arithmetic with p3/p3t.
    r3 = _ordered_arithmetic(t, [(2, p1), (2, p2), (2, p3)])
    lost3 = _ordered_arithmetic(r2, [(2, r3), (2, p3)])
    w3 = _sub(p3t, lost3)
    y03 = _sub(r3, w3)

    ix = _high_abs(t)
    exponent = (ix >> jnp.uint64(20)).astype(jnp.int64)
    expected_hw = jnp.where(n == 1, jnp.uint64(0x3FF921FB), jnp.uint64(0x400921FB))
    cancellation = ix == expected_hw
    loss1 = exponent - ((_high_abs(y01) >> jnp.uint64(20)) & jnp.uint64(0x7FF)).astype(jnp.int64)
    second = cancellation & (loss1 > 16)
    loss2 = exponent - ((_high_abs(y02) >> jnp.uint64(20)) & jnp.uint64(0x7FF)).astype(jnp.int64)
    third = second & (loss2 > 49)
    rr = jnp.where(third, r3, jnp.where(second, r2, r1))
    ww = jnp.where(third, w3, jnp.where(second, w2, w1))
    y0 = jnp.where(third, y03, jnp.where(second, y02, y01))
    y1 = _ordered_arithmetic(rr, [(2, y0), (2, ww)])
    return n, y0, y1


@jax.jit
def _sin_projection_angles(x):
    """Preserve old provider range; extend via source reduction to +/-pi."""
    if not jax.config.x64_enabled:
        raise ValueError("Requires binary64 mode")
    x = jnp.asarray(x, jnp.float64)
    ax = jnp.abs(x)
    low = _high_abs(x) < jnp.uint64(0x4002D97C)
    in_domain = ax <= jnp.pi
    t = jnp.where(in_domain & ~low, ax, jnp.asarray(3.0, jnp.float64))
    n, y0, y1 = _medium_reduce_abs(t)
    positive = jnp.where(n == 1, _kernel_cos(y0, y1), -_kernel_sin_tail(y0, y1))
    extended = jnp.where(jnp.signbit(x), -positive, positive)
    result = jnp.where(low, _old_sine(x), extended)
    return jnp.where(in_domain, result, jnp.nan)
