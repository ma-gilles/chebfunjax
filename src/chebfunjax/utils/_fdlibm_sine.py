"""Fdlibm sine for the bounded Chebyshev-angle interval.

Provenance
----------
Netlib fdlibm s_sin.c, k_sin.c, k_cos.c and e_rem_pio2.c (1993/1995).
Chebfun consumer: @chebtech2/chebpts.m; Chebfun commit: 7574c77.
Only the source high-word |x| < 3*pi/4 branch is implemented. This is not a
general sine replacement. A sequential arithmetic schedule preserves the source
rounding; optimization barriers alone are insufficient under compiled execution.

Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
Developed at SunSoft, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this software is freely
 granted, provided that this notice is preserved.
"""
import jax
import jax.numpy as jnp


def _add(a, b):
    return jax.lax.optimization_barrier(a + b)


def _sub(a, b):
    return jax.lax.optimization_barrier(a - b)


def _mul(a, b):
    return jax.lax.optimization_barrier(a * b)


def _high_abs(x):
    return (jax.lax.bitcast_convert_type(x, jnp.uint64) >> jnp.uint64(32)) & jnp.uint64(0x7fffffff)


def _ordered_arithmetic(initial, steps):
    """Execute each source binary operation in a separate loop iteration."""
    initial = jnp.asarray(initial, jnp.float64)
    operands = jnp.stack([jnp.broadcast_to(jnp.asarray(v, jnp.float64), initial.shape)
                          for _, v in steps])
    opcodes = jnp.asarray([op for op, _ in steps], jnp.int32)

    def step(value, instruction):
        opcode, operand = instruction
        result = jax.lax.switch(opcode, (
            lambda pair: pair[0] * pair[1],
            lambda pair: pair[0] + pair[1],
            lambda pair: pair[0] - pair[1],
        ), (value, operand))
        return result, None

    return jax.lax.scan(step, initial, (opcodes, operands), unroll=False)[0]


def _kernel_sin_zero_tail(x):
    s1 = -1.66666666666666324348e-01
    s2 = 8.33333333332248946124e-03
    s3 = -1.98412698298579493134e-04
    s4 = 2.75573137070700676789e-06
    s5 = -2.50507602534068634195e-08
    s6 = 1.58969099521155010221e-10
    z = _mul(x, x)
    v = _mul(z, x)
    r = _ordered_arithmetic(jnp.full_like(x, s6), [
        (0, z), (1, s5), (0, z), (1, s4), (0, z), (1, s3),
        (0, z), (1, s2), (0, z), (1, s1), (0, v), (1, x)])
    y = r
    return jnp.where(_high_abs(x) < 0x3e400000, x, y)


def _kernel_cos(x, y):
    c1 = 4.16666666666666019037e-02
    c2 = -1.38888888888741095749e-03
    c3 = 2.48015872894767294178e-05
    c4 = -2.75573143513906633035e-07
    c5 = 2.08757232129817482790e-09
    c6 = -1.13596475577881948265e-11
    ix = _high_abs(x)
    z = _mul(x, x)
    r = _ordered_arithmetic(jnp.full_like(x, c6), [
        (0, z), (1, c5), (0, z), (1, c4), (0, z), (1, c3),
        (0, z), (1, c2), (0, z), (1, c1), (0, z)])
    correction = _ordered_arithmetic(r, [(0, z), (2, _mul(x, y))])
    small = _sub(1., _ordered_arithmetic(z, [(0, .5), (2, correction)]))
    qbits = (ix - jnp.uint64(0x00200000)) << jnp.uint64(32)
    qx = jnp.where(ix > 0x3fe90000, .28125,
                   jax.lax.bitcast_convert_type(qbits, jnp.float64))
    hz = _ordered_arithmetic(z, [(0, .5), (2, qx)])
    large = _sub(_sub(1., qx), _sub(hz, correction))
    result = jnp.where(ix < 0x3fd33333, small, large)
    return jnp.where(ix < 0x3e400000, 1., result)


@jax.jit
def _sin_cheb_angles(x):
    """Source sine through the high-word |x| < 3*pi/4 branch; NaN outside.

    Provenance
    ----------
    Netlib fdlibm sine kernels and +/-pi/2 argument reduction.
    Chebfun consumer: @chebtech2/chebpts.m; Chebfun commit: 7574c77.
    """
    x = jnp.asarray(x, jnp.float64)
    ax = jnp.abs(x)
    ix = _high_abs(x)
    pio2_1 = 1.57079632673412561417e+00
    pio2_1t = 6.07710050650619224932e-11
    pio2_2 = 6.07710050630396597660e-11
    pio2_2t = 2.02226624879595063154e-21
    z = _sub(ax, pio2_1)
    ordinary_y0 = _sub(z, pio2_1t)
    ordinary_y1 = _sub(_sub(z, ordinary_y0), pio2_1t)
    near_z = _sub(z, pio2_2)
    near_y0 = _sub(near_z, pio2_2t)
    near_y1 = _sub(_sub(near_z, near_y0), pio2_2t)
    near = ix == 0x3ff921fb
    y0 = jnp.where(near, near_y0, ordinary_y0)
    y1 = jnp.where(near, near_y1, ordinary_y1)
    reduced = jnp.copysign(_kernel_cos(y0, y1), x)
    result = jnp.where(ix <= 0x3fe921fb, _kernel_sin_zero_tail(x), reduced)
    return jnp.where(ix < 0x4002d97c, result, jnp.nan)
