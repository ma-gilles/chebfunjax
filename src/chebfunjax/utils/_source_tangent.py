"""Finite source coefficient kernels for tiny complex TAN/TANH components.

Copyright (C) 1993,2004 Sun Microsystems, Inc. All rights reserved.
Permission to use, copy, modify, and distribute this software is freely granted,
provided that this notice is preserved. https://www.netlib.org/fdlibm/.
Existing general finite large reduction is reused unchanged. All newly
transcribed medium/special reduction arguments are gated to their source ranges.
"""

from functools import partial

import jax
import jax.numpy as j
import jax.numpy as jnp

from chebfunjax.utils._signed_gradual import (
    flip_sign_bits,
    gradual_signed_divide,
    gradual_signed_multiply,
)
from chebfunjax.utils._source_complex import source_expm1
from chebfunjax.utils._source_phase import _NPIO2_HW, source_large_rem_pio2
from chebfunjax.utils._source_qr._add import add_binary64
from chebfunjax.utils._source_real import add, div, mul, sub

T = (
    3.33333333333334091986e-1,
    1.33333333333201242699e-1,
    5.39682539762260521377e-2,
    2.18694882948595424599e-2,
    8.86323982359930005737e-3,
    3.59207910759131235356e-3,
    1.45620945432529025516e-3,
    5.88041240820264096874e-4,
    2.46463134818469906812e-4,
    7.81794442939557092300e-5,
    7.14072491382608190305e-5,
    -1.85586374855275456654e-5,
    2.59073051863633712884e-5,
)


def _word(x):
    return jax.lax.bitcast_convert_type(x, j.uint64)


def high(x):
    return ((_word(x) & j.uint64(0x7FFFFFFFFFFFFFFF)) >> j.uint64(32)).astype(j.int64)


def truncate(x):
    return jax.lax.bitcast_convert_type(_word(x) & j.uint64(0xFFFFFFFF00000000), j.float64)


def kernel(x, y, iy):
    ix = high(x)
    negative = (_word(x) & j.uint64(0x8000000000000000)) != 0

    def reciprocal(w, v):
        z = truncate(w)
        a = div(-1.0, w)
        t = truncate(a)
        s = add(1.0, mul(t, z))
        return add(t, mul(a, add(s, mul(t, v))))

    def tiny(_):
        def inverse(_):
            w = add(x, y)
            z = truncate(w)
            v = sub(y, sub(z, x))
            return reciprocal(w, v)

        return jax.lax.cond(
            iy == 1,
            lambda _: x,
            lambda _: jax.lax.cond(
                (_word(x) & j.uint64(0x7FFFFFFFFFFFFFFF)) == 0,
                lambda _: div(1.0, j.abs(x)),
                inverse,
                None,
            ),
            None,
        )

    def ordinary(_):
        upper = ix >= 0x3FE59428
        xx = j.where(upper, j.abs(x), x)
        yy = j.where(upper, j.where(negative, -y, y), y)
        z = sub(7.85398163397448278999e-1, xx)
        w = sub(3.06161699786838301793e-17, yy)
        xx = j.where(upper, add(z, w), xx)
        yy = j.where(upper, j.asarray(0.0, j.float64), yy)
        z = mul(xx, xx)
        w = mul(z, z)
        r = j.asarray(T[11], j.float64)
        for k in [9, 7, 5, 3, 1]:
            r = add(T[k], mul(w, r))
        v = j.asarray(T[12], j.float64)
        for k in [10, 8, 6, 4, 2]:
            v = add(T[k], mul(w, v))
        v = mul(z, v)
        s = mul(z, xx)
        r = add(yy, mul(z, add(mul(s, add(r, v)), yy)))
        r = add(r, mul(T[0], s))
        w = add(xx, r)

        def transformed(_):
            v = iy.astype(j.float64)
            answer = sub(v, mul(2.0, sub(xx, sub(div(mul(w, w), add(w, v)), r))))
            return j.where(negative, -answer, answer)

        def normal(_):
            def inverse(_):
                z = truncate(w)
                v = sub(r, sub(z, xx))
                return reciprocal(w, v)

            return jax.lax.cond(iy == 1, lambda _: w, inverse, None)

        return jax.lax.cond(upper, transformed, normal, None)

    return jax.lax.cond(ix < 0x3E300000, tiny, ordinary, None)


def reduce(x):
    ax = j.abs(x)
    ix = high(x)
    negative = (_word(x) & j.uint64(0x8000000000000000)) != 0

    def special(_):
        a = j.where((ix > 0x3FE921FB) & (ix < 0x4002D97C), ax, j.asarray(1.0, j.float64))
        z = sub(a, 1.57079632673412561417)
        near = ix == 0x3FF921FB
        z = j.where(near, sub(z, 6.07710050630396597660e-11), z)
        tail = j.where(near, 2.02226624879595063154e-21, 6.07710050650619224932e-11)
        hi = sub(z, tail)
        lo = sub(sub(z, hi), tail)
        return j.where(negative, -1, 1), j.where(negative, -hi, hi), j.where(negative, -lo, lo)

    def medium(_):
        t = j.where((ix >= 0x4002D97C) & (ix <= 0x413921FB), ax, j.asarray(3.0, j.float64))
        n = add(mul(t, 6.36619772367581382433e-1), 0.5).astype(j.int64)
        fn = n.astype(j.float64)
        r = sub(t, mul(fn, 1.57079632673412561417))
        w = mul(fn, 6.07710050650619224932e-11)
        hi = sub(r, w)

        def refine(state):
            r, w, hi = state
            loss = (ix >> 20) - ((high(hi) >> 20) & 2047)

            def second(state):
                temp, w, hi = state
                w = mul(fn, 6.07710050630396597660e-11)
                r = sub(temp, w)
                w = sub(mul(fn, 2.02226624879595063154e-21), sub(sub(temp, r), w))
                hi = sub(r, w)
                loss = (ix >> 20) - ((high(hi) >> 20) & 2047)

                def third(state):
                    temp, w, hi = state
                    w = mul(fn, 2.02226624871116645580e-21)
                    r = sub(temp, w)
                    w = sub(mul(fn, 8.47842766036889956997e-32), sub(sub(temp, r), w))
                    return r, w, sub(r, w)

                return jax.lax.cond(loss > 49, third, lambda s: s, (r, w, hi))

            return jax.lax.cond(loss > 16, second, lambda s: s, state)

        pattern = j.asarray(_NPIO2_HW, j.int64)[j.clip(n - 1, 0, 31)]
        r, w, hi = jax.lax.cond((n >= 32) | (ix == pattern), refine, lambda s: s, (r, w, hi))
        lo = sub(sub(r, hi), w)
        return j.where(negative, -n, n), j.where(negative, -hi, hi), j.where(negative, -lo, lo)

    def large(_):
        work = j.where((ix > 0x413921FB) & (ix < 0x7FF00000), x, j.asarray(2.0**20, j.float64))
        n, hi, lo, count = source_large_rem_pio2(work)
        return n, hi, lo

    return jax.lax.cond(
        ix < 0x4002D97C,
        special,
        lambda _: jax.lax.cond(ix <= 0x413921FB, medium, large, None),
        None,
    )


@jax.jit
def source_tan(x):
    x = j.asarray(x, j.float64)
    ix = high(x)

    def reduced(_):
        n, hi, lo = reduce(x)
        return kernel(hi, lo, 1 - ((n & 1) << 1))

    return jax.lax.cond(
        ix <= 0x3FE921FB,
        lambda _: kernel(x, j.asarray(0.0, j.float64), j.asarray(1, j.int64)),
        reduced,
        None,
    )


def _magnitude(value):
    return jax.lax.bitcast_convert_type(value, jnp.uint64) & jnp.uint64(0x7FFFFFFFFFFFFFFF)


@jax.jit
def source_tanh(x):
    a = jnp.abs(x)
    hi = (jax.lax.bitcast_convert_type(a, jnp.uint64) >> jnp.uint64(32)).astype(jnp.uint32)
    work = jnp.where(
        (hi >= jnp.uint32(0x3C800000)) & (hi < jnp.uint32(0x40360000)),
        a,
        jnp.asarray(1.0, jnp.float64),
    )

    def small(_):
        def central(_):
            t = source_expm1(mul(-2.0, work))
            return div(-t, add(t, 2.0))

        def upper(_):
            t = source_expm1(mul(2.0, work))
            return sub(1.0, div(2.0, add(t, 2.0)))

        return jax.lax.cond(hi < jnp.uint32(0x3FF00000), central, upper, None)

    mag = jax.lax.cond(
        hi < jnp.uint32(0x3C800000),
        lambda _: a,
        lambda _: jax.lax.cond(
            hi < jnp.uint32(0x40360000), small, lambda _: jnp.asarray(1.0, jnp.float64), None
        ),
        None,
    )
    return jnp.copysign(mag, x)


def _tiny_quotient_component(v, t, minus):
    p = gradual_signed_multiply(v, t)
    q = gradual_signed_multiply(p, t)
    square = gradual_signed_multiply(p, p)
    den, _ = add_binary64(jnp.ones_like(p), square)
    num, _ = add_binary64(v, flip_sign_bits(q) if minus else q)
    wide = gradual_signed_divide(num, den)
    pm = _magnitude(p)
    narrow_selected = pm > jnp.uint64(0x3FF0000000000000)
    safe = jnp.where(narrow_selected, p, jnp.asarray(2.0, jnp.float64))
    ratio = gradual_signed_divide(jnp.ones_like(p), safe)
    nd, _ = add_binary64(safe, ratio)
    vp = gradual_signed_multiply(v, ratio)
    nn, _ = add_binary64(vp, flip_sign_bits(t)) if minus else add_binary64(t, vp)
    narrow = gradual_signed_divide(nn, nd)
    vh = gradual_signed_multiply(v, jnp.asarray(0.5, jnp.float64))
    th = gradual_signed_multiply(t, jnp.copysign(jnp.full_like(p, 0.5), p))
    equal, _ = add_binary64(vh, flip_sign_bits(th) if minus else th)
    return jnp.where(
        narrow_selected, narrow, jnp.where(pm == jnp.uint64(0x3FF0000000000000), equal, wide)
    )


@partial(jax.jit, static_argnames=("hyperbolic",))
def tiny_component_words(real, imag, prior_real, prior_imag, hyperbolic):
    """Literal finite Smith target components for nonzero identity-branch inputs.

    TAN=(tan(x)+i*tanh(y))/(1-i*tan(x)*tanh(y)); TANH swaps
    the real functions and the denominator sign. For 0<|component|<2^-55 (the canonical fdlibm TANH identity branch),
    both real TAN/TANH are exactly the input. Keep both product roundings,
    the Smith denominator/branch, and the final divide. Captured native
    operands are independent tests only; no runtime native buffers.
    """
    if real.size == 0:
        return prior_real, prior_imag
    mr, mi = _magnitude(real), _magnitude(imag)
    finite = (mr < jnp.uint64(0x7FF0000000000000)) & (mi < jnp.uint64(0x7FF0000000000000))
    real_selected = finite & (mr > 0) & (mr < jnp.uint64(0x3C80000000000000))
    imag_selected = finite & (mi > 0) & (mi < jnp.uint64(0x3C80000000000000))
    real_fn = source_tan if hyperbolic else source_tanh
    imag_fn = source_tanh if hyperbolic else source_tan
    real_coefficient = jax.lax.map(
        real_fn, jnp.where(real_selected, imag, 0.0).reshape(-1)
    ).reshape(real.shape)
    imag_coefficient = jax.lax.map(
        imag_fn, jnp.where(imag_selected, real, 0.0).reshape(-1)
    ).reshape(imag.shape)
    real_selected = real_selected & (_magnitude(real_coefficient) < jnp.uint64(0x7FF0000000000000))
    imag_selected = imag_selected & (_magnitude(imag_coefficient) < jnp.uint64(0x7FF0000000000000))
    rr = _tiny_quotient_component(
        jnp.where(real_selected, real, 0.0),
        jnp.where(real_selected, real_coefficient, 0.0),
        not hyperbolic,
    )
    ii = _tiny_quotient_component(
        jnp.where(imag_selected, imag, 0.0),
        jnp.where(imag_selected, imag_coefficient, 0.0),
        hyperbolic,
    )
    return (
        jnp.where(real_selected, jax.lax.bitcast_convert_type(rr, jnp.uint64), prior_real),
        jnp.where(imag_selected, jax.lax.bitcast_convert_type(ii, jnp.uint64), prior_imag),
    )


@partial(jax.jit, static_argnames=("hyperbolic",))
def finite_tangent_result(value, result, hyperbolic):
    """Cache the identical finite tiny-component conditional across eager calls."""
    real, imag = jnp.real(value), jnp.imag(value)
    prior = jax.lax.optimization_barrier(result)
    real_words = jax.lax.bitcast_convert_type(real, jnp.uint64)
    imag_words = jax.lax.bitcast_convert_type(imag, jnp.uint64)
    magnitude = jnp.uint64(0x7FFFFFFFFFFFFFFF)
    mr, mi = real_words & magnitude, imag_words & magnitude
    finite = (mr < jnp.uint64(0x7FF0000000000000)) & (mi < jnp.uint64(0x7FF0000000000000))
    tiny = finite & (
        ((mr > 0) & (mr < jnp.uint64(0x3C80000000000000)))
        | ((mi > 0) & (mi < jnp.uint64(0x3C80000000000000)))
    )

    def source_components(_):
        rr = jax.lax.optimization_barrier(jax.lax.bitcast_convert_type(jnp.real(prior), jnp.uint64))
        ii = jax.lax.optimization_barrier(jax.lax.bitcast_convert_type(jnp.imag(prior), jnp.uint64))
        rr, ii = tiny_component_words(real, imag, rr, ii, hyperbolic)
        return jax.lax.complex(
            jax.lax.bitcast_convert_type(rr, jnp.float64),
            jax.lax.bitcast_convert_type(ii, jnp.float64),
        )

    result = jax.lax.cond(jnp.any(tiny), source_components, lambda _: prior, None)
    return result
