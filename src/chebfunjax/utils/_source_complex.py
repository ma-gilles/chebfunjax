"""Source scalar arithmetic used by the small Hermite ASY path.

Provenance: Chebfun hermpts.m, commit
7574c77680d7e82b79626300bf255498271a72df; AMOS Airy/Bessel algorithms
by Donald E. Amos; Netlib fdlibm e_pow.c, e_exp.c, e_log.c,
e_rem_pio2.c and k_rem_pio2.c.

Fdlibm copyright (C) 1993, 2004 Sun Microsystems, Inc. All rights reserved.
Developed at SunSoft, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this software is freely
granted, provided that this notice is preserved.

These are private ordinary-domain source kernels, not complete replacements
for public transcendental APIs. Static mathematical coefficients are kept
in the repository. No captured operands, output tables or runtime fixtures.
"""

from functools import partial

import jax
import jax.numpy as j
import jax.numpy as jnp
from jax import lax

from ._source_phase import source_medium_phase_jit, source_phase_jit
from ._source_real import add, div, mul, source_exp_jit, source_log_jit, source_pow_jit, sub
from ._source_scalar_constants import ATAN as _ATAN
from ._source_scalar_constants import EXPM1 as _EXPM1


@partial(jax.jit, static_argnames=("operation",))
def eager_binary(a, b, operation):
    """Retain a source eager-operation boundary inside a traced computation.

    Explicit source complex products use ``ordered_multiply`` below. This
    boundary preserves ordinary JAX complex arithmetic within a single
    operation, including the source transcription's existing contraction.
    """
    functions = (j.multiply, j.add, j.subtract, j.divide)
    generic = functions[operation]
    a, b = j.asarray(a), j.asarray(b)
    dtype = j.result_type(a, b)
    a, b = j.broadcast_arrays(a.astype(dtype), b.astype(dtype))
    if dtype == j.float64:
        from ._source_binary import binary

        return binary(a, b, operation)
    if dtype != j.complex128:
        return generic(a, b)

    def ordered(a, b):
        operands = j.stack((b, j.ones_like(b)))
        identities = j.asarray([False, True])

        def step(value, instruction):
            is_identity, operand = instruction

            # A complex multiply by 1+0j can change the sign of an exact
            # zero component. Materialize each component with real *1.
            def identity():
                return lax.complex(value.real * operand.real, value.imag * operand.real)

            value = lax.cond(is_identity, identity, lambda: generic(value, operand))
            return value, None

        return lax.scan(step, a, (identities, operands), unroll=False)[0]

    def cpu(a, b):
        finite = j.all(j.isfinite(a)) & j.all(j.isfinite(b))
        return lax.cond(finite, ordered, generic, a, b)

    return lax.platform_dependent(a, b, cpu=cpu, default=generic)


def ordered_multiply(a, b):
    """Materialize the four products and both sums used by the ASY source."""
    from ._source_binary import binary

    a, b = j.asarray(a, j.complex128), j.asarray(b, j.complex128)
    left = j.stack((a.real, a.imag, a.real, a.imag))
    right = j.stack((b.real, b.imag, b.imag, b.real))
    products = binary(left, right, 0)
    return lax.complex(binary(products[0], products[1], 2), binary(products[2], products[3], 1))


def source_complex_divide(a, b):
    """Finite source Smith quotient, including real and imaginary axes."""
    a, b = j.asarray(a, j.complex128), j.asarray(b, j.complex128)
    ar, ai, br, bi = a.real, a.imag, b.real, b.imag
    zero = j.asarray(0.0, j.float64)

    def wide(_):
        ratio = div(bi, br)
        den = add(br, mul(bi, ratio))
        return lax.complex(div(add(ar, mul(ai, ratio)), den), div(sub(ai, mul(ar, ratio)), den))

    def equal(_):
        r, s = j.copysign(j.asarray(0.5), br), j.copysign(j.asarray(0.5), bi)
        den = j.abs(br)
        return lax.complex(
            div(add(mul(ar, r), mul(ai, s)), den), div(sub(mul(ai, r), mul(ar, s)), den)
        )

    def narrow(_):
        ratio = div(br, bi)
        den = add(bi, mul(br, ratio))
        return lax.complex(div(add(ai, mul(ar, ratio)), den), div(sub(mul(ai, ratio), ar), den))

    def general(_):
        return lax.cond(
            j.abs(br) > j.abs(bi),
            wide,
            lambda _: lax.cond(j.abs(br) == j.abs(bi), equal, narrow, None),
            None,
        )

    def real_axis(_):
        return lax.complex(
            lax.cond(ar != 0.0, lambda _: div(ar, br), lambda _: zero, None),
            lax.cond(ai != 0.0, lambda _: div(ai, br), lambda _: zero, None),
        )

    def imag_axis(_):
        return lax.complex(
            lax.cond(ai != 0.0, lambda _: div(ai, bi), lambda _: zero, None),
            lax.cond(ar != 0.0, lambda _: -div(ar, bi), lambda _: zero, None),
        )

    return lax.cond(
        bi == 0.0, real_axis, lambda _: lax.cond(br == 0.0, imag_axis, general, None), None
    )


def source_scale_divide(a, b):
    """Source real/complex derivative-scale route, without axis shortcuts."""
    br, bi = b.real, b.imag

    def wide(_):
        ratio = div(bi, br)
        den = add(br, mul(bi, ratio))
        return lax.complex(div(a, den), div(-mul(a, ratio), den))

    def narrow(_):
        ratio = div(br, bi)
        den = add(bi, mul(br, ratio))
        return lax.complex(div(mul(a, ratio), den), div(-a, den))

    return lax.cond(j.abs(br) > j.abs(bi), wide, narrow, None)


@partial(jax.jit, static_argnames=("n",))
def source_integer_power(z, n):
    z = j.asarray(z, j.complex128)

    def complex_power(_):
        result, base = j.asarray(1.0 + 0j, j.complex128), z
        count = n
        while count:
            if count % 2:
                result = ordered_multiply(base, result)
            base = ordered_multiply(base, base)
            count //= 2
        return result

    return lax.cond(
        z.imag == 0.0,
        lambda _: lax.complex(source_pow_jit(z.real, j.asarray(float(n))), j.asarray(0.0)),
        complex_power,
        None,
    )


def _high(x):
    return (lax.bitcast_convert_type(j.abs(x), j.uint64) >> j.uint64(32)).astype(j.int64)


def source_atan(x):
    x = j.asarray(x, j.float64)
    ix, original = _high(x), x

    def polynomial(x):
        z, w = mul(x, x), mul(mul(x, x), mul(x, x))
        p = add(mul(_ATAN[0x337EFC0], w), _ATAN[0x337EFC8])
        for address in (0x337EFD0, 0x337EFD8, 0x337EFE0, 0x337EFE8):
            p = add(mul(p, w), _ATAN[address])
        p = mul(p, z)
        q = sub(mul(_ATAN[0x337EFF0], w), _ATAN[0x337EFF8])
        for address in (0x337F000, 0x337F008, 0x337F010):
            q = sub(mul(q, w), _ATAN[address])
        return mul(add(p, mul(q, w)), x)

    def central(_):
        return lax.cond(ix <= 0x3E1FFFFF, lambda _: x, lambda _: sub(x, polynomial(x)), None)

    def reduced(_):
        a = j.abs(x)

        def first(_):
            return (
                div(sub(add(a, a), 1.0), add(a, 2.0)),
                j.asarray(_ATAN[0x337EFA8]),
                j.asarray(_ATAN[0x337EFB0]),
            )

        def second(_):
            return (
                div(sub(a, 1.0), add(a, 1.0)),
                j.asarray(_ATAN[0x337EC58]),
                j.asarray(_ATAN[0x337EC50]),
            )

        def third(_):
            return (
                div(sub(a, 1.5), add(1.0, mul(1.5, a))),
                j.asarray(_ATAN[0x337EF98]),
                j.asarray(_ATAN[0x337EFA0]),
            )

        def last(_):
            return div(-1.0, a), j.asarray(_ATAN[0x337EF90]), j.asarray(_ATAN[0x337EF28])

        r, lo, hi = lax.cond(
            ix <= 0x3FE5FFFF,
            first,
            lambda _: lax.cond(
                ix <= 0x3FF2FFFF,
                second,
                lambda _: lax.cond(ix <= 0x40037FFF, third, last, None),
                None,
            ),
            None,
        )
        return j.copysign(sub(hi, sub(sub(polynomial(r), lo), r)), original)

    return lax.cond(
        ix > 0x440FFFFF,
        lambda _: j.copysign(j.asarray(_ATAN[0x337EF28]), x),
        lambda _: lax.cond(ix <= 0x3FDBFFFF, central, reduced, None),
        None,
    )


def source_atan2(y, x):
    x, y = j.asarray(x, j.float64), j.asarray(y, j.float64)
    sx, sy = j.signbit(x), j.signbit(y)
    quadrant = 2 * sx.astype(j.int32) + sy.astype(j.int32)

    def ordinary(_):
        diff = _high(y) - _high(x)
        z = lax.cond(
            diff > 0x3CFFFFF,
            lambda _: j.asarray(_ATAN[0x337EF28]),
            lambda _: lax.cond(
                sx & ((diff >> 20) < -60),
                lambda _: j.asarray(0.0),
                lambda _: source_atan(j.abs(div(y, x))),
                None,
            ),
            None,
        )
        return lax.switch(
            quadrant,
            (
                lambda: z,
                lambda: -z,
                lambda: sub(_ATAN[0x337EF30], sub(z, _ATAN[0x337F028])),
                lambda: sub(sub(z, _ATAN[0x337F028]), _ATAN[0x337EF30]),
            ),
        )

    def y_zero(_):
        return lax.cond(
            x == 0.0,
            lambda _: j.asarray(0.0),
            lambda _: j.where(quadrant < 2, y, j.copysign(j.asarray(_ATAN[0x337EF30]), y)),
            None,
        )

    return lax.cond(
        x == 1.0,
        lambda _: source_atan(y),
        lambda _: lax.cond(
            y == 0.0,
            y_zero,
            lambda _: lax.cond(
                x == 0.0, lambda _: j.copysign(j.asarray(_ATAN[0x337EF28]), y), ordinary, None
            ),
            None,
        ),
        None,
    )


def asinh_source(x):
    a, hi = j.abs(x), _high(x)

    def ordinary(_):
        t = mul(x, x)
        y = lax.cond(
            hi > 0x40000000,
            lambda _: source_log_jit(add(add(a, a), div(1.0, add(j.sqrt(add(t, 1.0)), a)))),
            lambda _: j.log1p(add(a, div(t, add(1.0, j.sqrt(add(1.0, t)))))),
            None,
        )
        return j.copysign(y, x)

    return lax.cond(
        hi <= 0x3E2FFFFF,
        lambda _: x,
        lambda _: lax.cond(
            hi > 0x41B00000,
            lambda _: j.copysign(add(source_log_jit(a), float.fromhex("0x1.62e42fefa39efp-1")), x),
            ordinary,
            None,
        ),
        None,
    )


def real_acos_source(x):
    x = j.asarray(x, j.float64)

    def continued(_):
        p, m = add(1.0, x), sub(1.0, x)
        pr, pi = j.sqrt(j.maximum(p, 0.0)), j.sqrt(j.maximum(-p, 0.0))
        mr, mi = j.sqrt(j.maximum(m, 0.0)), j.sqrt(j.maximum(-m, 0.0))
        return lax.complex(mul(2.0, j.arctan2(mr, pr)), asinh_source(sub(mul(pr, mi), mul(mr, pi))))

    return lax.cond(
        j.abs(x) <= 1.0, lambda _: lax.complex(j.arccos(x), j.asarray(0.0)), continued, None
    )


@partial(jax.jit, static_argnames=("exponent",))
def principal_power(z, exponent):
    if exponent not in (2 / 3, 0.25):
        raise ValueError("source principal power supports exponents 2/3 and 1/4")
    z = j.asarray(z, j.complex128)
    a, b = z.real, z.imag

    def continued(_):
        norm = _native_hypot(a, b)

        def near_unit(_):
            aa, bb = mul(a, a), mul(b, b)
            delta = lax.cond(
                aa > bb, lambda _: -sub(sub(1.0, aa), bb), lambda _: -sub(sub(1.0, bb), aa), None
            )
            return mul(0.5, j.log1p(delta))

        lr = lax.cond(
            (norm > 0.875) & (norm < 1.125), near_unit, lambda _: source_log_jit(norm), None
        )
        phase = mul(source_atan2(b, a), exponent)
        # atan2 is in [-pi, pi]; these source exponents bound this phase by
        # 2*pi/3. Use the identical medium kernel without a large graph.
        sine, cosine = source_medium_phase_jit(phase)
        radius = source_exp_jit(mul(lr, exponent))
        return lax.complex(mul(radius, cosine), mul(radius, sine))

    return lax.cond(
        (b == 0.0) & (a >= 0.0),
        lambda _: lax.complex(source_pow_jit(a, exponent), j.copysign(j.asarray(0.0), b)),
        continued,
        None,
    )


def _scale(y, k):
    bits = lax.bitcast_convert_type(y, j.uint64) + (k.astype(j.uint64) << j.uint64(52))
    return lax.bitcast_convert_type(bits, j.float64)


def source_expm1(original):
    original = j.asarray(original, j.float64)
    hx = _high(original)
    k = j.where(
        hx <= 0x3FF0A2B1,
        j.where(original > 0.0, 1, -1),
        j.trunc(add(mul(original, _EXPM1[0x337ED38]), j.where(original > 0.0, 0.5, -0.5))).astype(
            j.int64
        ),
    )
    k = j.where(hx > 0x3FD62E42, k, j.asarray(0, j.int64))
    hi = sub(original, mul(k.astype(j.float64), _EXPM1[0x337ED40]))
    lo = mul(k.astype(j.float64), _EXPM1[0x337ED28])
    x = sub(hi, lo)
    c = sub(sub(hi, x), lo)
    hfx = mul(0.5, x)
    hxs = mul(x, hfx)
    r = add(mul(_EXPM1[0x337ED48], hxs), _EXPM1[0x337ED50])
    r = sub(mul(r, hxs), _EXPM1[0x337ED58])
    r = add(mul(r, hxs), _EXPM1[0x337ED60])
    r = sub(mul(r, hxs), _EXPM1[0x337ED68])
    r = add(mul(r, hxs), 1.0)
    t = sub(3.0, mul(hfx, r))
    e = mul(hxs, div(sub(r, t), sub(6.0, mul(x, t))))
    base = sub(x, sub(mul(x, e), hxs))
    e = sub(sub(mul(x, sub(e, c)), c), hxs)

    def shifted(_):
        def middle(_):
            t = lax.bitcast_convert_type(
                (j.asarray(0x3FF00000, j.int64) - (j.asarray(0x200000, j.int64) >> k)).astype(
                    j.uint64
                )
                << j.uint64(32),
                j.float64,
            )
            return _scale(sub(t, sub(e, x)), k)

        def large(_):
            t = lax.bitcast_convert_type(
                (j.asarray(1023, j.int64) - k).astype(j.uint64) << j.uint64(52), j.float64
            )
            return _scale(add(sub(x, add(e, t)), 1.0), k)

        return lax.cond(
            (k <= -2) | (k > 56),
            lambda _: sub(_scale(sub(1.0, sub(e, x)), k), 1.0),
            lambda _: lax.cond(k < 20, middle, large, None),
            None,
        )

    result = lax.cond(
        k == 0,
        lambda _: base,
        lambda _: lax.cond(
            k == -1,
            lambda _: sub(mul(0.5, sub(x, e)), 0.5),
            lambda _: lax.cond(
                k == 1,
                lambda _: lax.cond(
                    x < -0.25,
                    lambda _: mul(-2.0, sub(e, add(x, 0.5))),
                    lambda _: add(1.0, mul(2.0, sub(x, e))),
                    None,
                ),
                shifted,
                None,
            ),
            None,
        ),
        None,
    )
    return j.where(hx <= 0x3C8FFFFF, original, result)


def source_cosh(x):
    a, hx = j.abs(x), _high(x)

    def central(_):
        t = source_expm1(a)
        w = add(1.0, t)
        return lax.cond(
            hx <= 0x3C7FFFFF, lambda _: w, lambda _: add(1.0, div(mul(t, t), add(w, w))), None
        )

    def outer(_):
        t = source_exp_jit(a)
        return add(mul(0.5, t), div(0.5, t))

    return lax.cond(hx <= 0x3FD62E42, central, outer, None)


def source_sinh(x):
    a, hx = j.abs(x), _high(x)

    def ordinary(_):
        h, t = j.copysign(j.asarray(0.5), x), source_expm1(a)
        return lax.cond(
            hx <= 0x3FEFFFFF,
            lambda _: mul(h, sub(add(t, t), div(mul(t, t), add(t, 1.0)))),
            lambda _: mul(h, add(t, div(t, add(t, 1.0)))),
            None,
        )

    return lax.cond(hx <= 0x3E2FFFFF, lambda _: x, ordinary, None)


def complex_trig(z):
    z = j.asarray(z, j.complex128)
    sine, cosine = source_phase_jit(z.real)
    zero = j.asarray(0.0)

    def continued(_):
        ch, sh = source_cosh(z.imag), source_sinh(z.imag)
        return lax.cond(
            z.real == 0.0,
            lambda _: (lax.complex(ch, zero), lax.complex(zero, sh)),
            lambda _: (
                lax.complex(mul(cosine, ch), -mul(sine, sh)),
                lax.complex(mul(sine, ch), mul(cosine, sh)),
            ),
            None,
        )

    return lax.cond(
        z.imag == 0.0,
        lambda _: (lax.complex(cosine, zero), lax.complex(sine, zero)),
        continued,
        None,
    )


@jax.jit
def source_real_div(a, z):
    a = j.asarray(a, j.float64)
    z = j.asarray(z, j.complex128)
    (br, bi) = (z.real, z.imag)
    c = jax.lax.complex
    zero = j.asarray(0.0, j.float64)

    def general(_):

        def wide(_):
            r = div(bi, br)
            den = add(br, mul(bi, r))
            return c(div(a, den), div(-mul(a, r), den))

        def equal(_):
            den = j.abs(br)
            return c(
                div(mul(a, j.copysign(j.asarray(0.5), br)), den),
                div(-mul(a, j.copysign(j.asarray(0.5), bi)), den),
            )

        def narrow(_):
            r = div(br, bi)
            den = add(bi, mul(br, r))
            return c(div(mul(a, r), den), div(-a, den))

        return jax.lax.cond(
            j.abs(br) > j.abs(bi),
            wide,
            lambda _: jax.lax.cond(j.abs(br) == j.abs(bi), equal, narrow, None),
            None,
        )

    return jax.lax.cond(
        bi == 0.0,
        lambda _: c(div(a, br), zero),
        lambda _: jax.lax.cond(br == 0.0, lambda _: c(zero, div(-a, bi)), general, None),
        None,
    )


def _complex(real, imag):
    return lax.complex(jnp.asarray(real, dtype=jnp.float64), jnp.asarray(imag, dtype=jnp.float64))


def _native_hypot(x, y):
    (a, b) = (jnp.maximum(jnp.abs(x), jnp.abs(y)), jnp.minimum(jnp.abs(x), jnp.abs(y)))

    def high(v):
        return lax.bitcast_convert_type(v, jnp.uint64) >> jnp.uint64(32)

    def upper(v):
        return lax.bitcast_convert_type(
            lax.bitcast_convert_type(v, jnp.uint64) & jnp.uint64(18446744069414584320), jnp.float64
        )

    (ha, hb) = (high(a), high(b))

    def ordinary(_):
        exponent = jnp.where(
            ha > 1596981248, -600, jnp.where(hb < 548405248, jnp.where(hb < 1048576, 1022, 600), 0)
        )
        (aa, bb) = (jnp.ldexp(a, exponent), jnp.ldexp(b, exponent))
        w = aa - bb

        def wide(_):
            t1 = upper(aa)
            t2 = aa - t1
            left = jnp.stack((t1, bb, t2))
            right = jnp.stack((t1, -bb, aa + t1))
            p = lax.map(lambda q: q[0] * q[1], jnp.stack((left, right), axis=-1))
            return jnp.sqrt(p[0] - (p[1] - p[2]))

        def close(_):
            a2 = aa + aa
            t1 = upper(a2)
            t2 = a2 - t1
            y1 = upper(bb)
            y2 = bb - y1
            left = jnp.stack((t1, w, t1, t2))
            right = jnp.stack((y1, -w, y2, bb))
            p = lax.map(lambda q: q[0] * q[1], jnp.stack((left, right), axis=-1))
            return jnp.sqrt(p[0] - (p[1] - (p[2] + p[3])))

        return jnp.ldexp(lax.cond(w > bb, wide, close, None), -exponent)

    return lax.cond(
        (b == 0.0) | (ha - hb > 62914560) | ~jnp.isfinite(a), lambda _: a + b, ordinary, None
    )


def _native_moderate_sqrt(z):
    (ar, ai) = (z.real, jnp.where(z.imag == 0.0, jnp.asarray(0.0, jnp.float64), z.imag))
    radius = _native_hypot(ar, ai)
    magnitude = jnp.sqrt(0.5 * radius + 0.5 * jnp.abs(ar))
    other = 0.5 * (ai / magnitude)
    result = jnp.where(
        ar > 0.0, _complex(magnitude, other), _complex(jnp.abs(other), jnp.copysign(magnitude, ai))
    )
    result = jnp.where(
        (ar == 0.0) & (ai != 0.0), _complex(magnitude, jnp.copysign(magnitude, ai)), result
    )
    return jnp.where((ai == 0.0) & (ar >= 0.0), _complex(jnp.sqrt(ar), 0.0), result)
